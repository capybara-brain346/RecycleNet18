from typing import TypedDict, List, Set, Union
from datetime import datetime
import json
import time
from decimal import Decimal

from api.services.dataset_service import DatasetService
from api.services.training_service import TrainingService
from api.services.model_service import ModelService
from api.services.inference_service import InferenceService
from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq


class AgentState(TypedDict):
    dataset_service: DatasetService
    training_service: TrainingService
    model_service: ModelService
    inference_service: InferenceService
    seen_datasets: Set[str]
    evaluated_jobs: Set[str]
    processed_logs: Set[str]
    new_datasets: List[str]
    messages: List[str]


def ask_groq_llm(prompt: str):
    llm = ChatGroq(
        temperature=0,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    response = llm.invoke(prompt)
    return response.content


def detect_new_datasets(state: AgentState) -> AgentState:
    print("[Training-Trigger Agent] Checking for new annotated datasets...")

    dataset_service = state["dataset_service"]
    seen_datasets = state["seen_datasets"]
    datasets = dataset_service.list_datasets(data_type="annotated")

    dataset_keys = [d["key"] for d in datasets if isinstance(d, dict)]
    new_datasets = [d for d in dataset_keys if d not in seen_datasets]

    if new_datasets:
        state["messages"].append(
            f"Found {len(new_datasets)} new datasets: {new_datasets}"
        )
    state["new_datasets"] = new_datasets
    return state


def train_on_new_datasets(state: AgentState) -> AgentState:
    print("[Training-Trigger Agent] Processing new datasets for training...")

    training_service = state["training_service"]
    new_datasets = state.get("new_datasets", [])

    for dataset in new_datasets:
        prompt = (
            f"A new annotated dataset is available for YOLOv8 training: {dataset}. "
            "Based on MLOps best practices, suggest optimal hyperparameters as a JSON dictionary. "
            "Include: epochs (int), batch_size (int), learning_rate (float), and any other relevant YOLOv8 parameters. "
            "Only output valid JSON."
        )
        try:
            llm_response = ask_groq_llm(prompt)
            llm_response = llm_response.strip()
            if not llm_response.startswith("{"):
                start_idx = llm_response.find("{")
                end_idx = llm_response.rfind("}") + 1
                if start_idx != -1 and end_idx != 0:
                    llm_response = llm_response[start_idx:end_idx]
                else:
                    raise ValueError("No valid JSON found in LLM response")

            hyperparameters = json.loads(llm_response)

            for key, value in hyperparameters.items():
                if isinstance(value, float):
                    hyperparameters[key] = Decimal(str(value))

            state["messages"].append(
                f"Starting training on dataset {dataset} with hyperparameters: {hyperparameters}"
            )
            job_id = training_service.start_training(
                dataset_id=dataset, hyperparameters=hyperparameters
            )
            state["seen_datasets"].add(dataset)
        except (json.JSONDecodeError, ValueError) as e:
            state["messages"].append(
                f"Error parsing hyperparameters for dataset {dataset}: {str(e)}. Using default hyperparameters."
            )
            default_hyperparameters = {
                "epochs": 100,
                "batch_size": 16,
                "learning_rate": Decimal("0.01"),
                "image_size": 640,
                "patience": 50,
                "device": "cuda",
            }
            job_id = training_service.start_training(
                dataset_id=dataset, hyperparameters=default_hyperparameters
            )
            state["seen_datasets"].add(dataset)
        except Exception as e:
            state["messages"].append(f"Error processing dataset {dataset}: {str(e)}")
            continue

    return state


def evaluate_and_promote(state: AgentState) -> AgentState:
    print("[Model-Evaluator Agent] Evaluating completed training jobs...")

    training_service = state["training_service"]
    model_service = state["model_service"]
    evaluated_jobs = state["evaluated_jobs"]

    jobs = training_service.list_jobs()
    completed_jobs = [
        j
        for j in jobs
        if j.get("status") == "completed"
        and j.get("training_jobs_partition") not in evaluated_jobs
    ]

    for job in completed_jobs:
        job_id = job.get("training_jobs_partition")
        if not job_id:
            continue

        job_details = training_service.get_job_status(job_id)
        if not job_details:
            continue

        new_metrics = job_details.get("metrics", [])
        if not new_metrics:
            state["messages"].append(
                f"Skipping evaluation for job {job_id}: No metrics available"
            )
            evaluated_jobs.add(job_id)
            continue

        latest_metrics = (
            new_metrics[-1] if isinstance(new_metrics, list) else new_metrics
        )

        prod_model = model_service.get_production_model()
        prod_metrics = prod_model.get("metrics", []) if prod_model else []
        latest_prod_metrics = prod_metrics[-1] if prod_metrics else {}

        new_metrics_display = {
            "mAP50-95": float(latest_metrics.get("mAP50-95", 0)),
            "mAP50": float(latest_metrics.get("mAP50", 0)),
            "precision": float(latest_metrics.get("precision", 0)),
            "recall": float(latest_metrics.get("recall", 0)),
        }

        prod_metrics_display = {
            "mAP50-95": float(latest_prod_metrics.get("mAP50-95", 0)),
            "mAP50": float(latest_prod_metrics.get("mAP50", 0)),
            "precision": float(latest_prod_metrics.get("precision", 0)),
            "recall": float(latest_prod_metrics.get("recall", 0)),
        }

        prompt = (
            f"Compare these two YOLOv8 models based on their metrics:\n"
            f"New model metrics: {json.dumps(new_metrics_display, indent=2)}\n"
            f"Current production model metrics: {json.dumps(prod_metrics_display, indent=2)}\n"
            "Should we promote the new model to production? Consider:\n"
            "1. mAP50-95 improvement (must be >1% better)\n"
            "2. Inference speed\n"
            "3. Model size and resource requirements\n"
            "Reply with exactly 'yes' or 'no' followed by a brief explanation."
        )

        llm_response = ask_groq_llm(prompt)
        state["messages"].append(f"Model evaluation for {job_id}: {llm_response}")

        if llm_response.strip().lower().startswith("yes"):
            state["messages"].append(
                f"Promoting model {job_id} to production based on superior metrics."
            )
            model_service.promote_model(model_id=job_id)

        evaluated_jobs.add(job_id)

    return state


def active_learning(state: AgentState) -> AgentState:
    print(
        "[Active-Learning Agent] Analyzing inference logs for uncertain predictions..."
    )

    inference_service = state["inference_service"]
    dataset_service = state["dataset_service"]
    processed_logs = state["processed_logs"]

    logs = inference_service.get_inference_logs()
    new_logs = [log for log in logs if log.get("id") not in processed_logs]

    for log in new_logs:
        log_id = log.get("id")
        predictions = log.get("predictions", [])
        uncertain_preds = [
            pred for pred in predictions if pred.get("confidence", 1.0) < 0.3
        ]

        if uncertain_preds:
            state["messages"].append(
                f"Found {len(uncertain_preds)} uncertain predictions in log {log_id}"
            )

            for pred in uncertain_preds:
                image_path = pred.get("image_path")
                if not image_path:
                    continue

                confidence = pred.get("confidence", 1.0)
                prompt = (
                    f"Analyzing prediction confidence of {confidence} for image {image_path}.\n"
                    "Given our active learning strategy:\n"
                    "1. Confidence < 0.3 indicates high uncertainty\n"
                    "2. We want to focus on edge cases\n"
                    "Should this image be flagged for expert re-annotation?\n"
                    "Reply with exactly 'yes' or 'no' followed by reasoning."
                )

                llm_response = ask_groq_llm(prompt)
                if llm_response.strip().lower().startswith("yes"):
                    dataset_service.flag_for_reannotation(image_path)
                    state["messages"].append(
                        f"Flagged {image_path} for re-annotation due to low confidence: {confidence}"
                    )

        processed_logs.add(log_id)

    return state


def main():
    initial_state: AgentState = {
        "dataset_service": DatasetService(),
        "training_service": TrainingService(),
        "model_service": ModelService(),
        "inference_service": InferenceService(),
        "seen_datasets": set(),
        "evaluated_jobs": set(),
        "processed_logs": set(),
        "new_datasets": [],
        "messages": [],
    }

    workflow = StateGraph(AgentState)

    workflow.add_node("detect_new_datasets", detect_new_datasets)
    workflow.add_node("train_on_new_datasets", train_on_new_datasets)
    workflow.add_node("evaluate_and_promote", evaluate_and_promote)
    workflow.add_node("active_learning", active_learning)

    workflow.add_edge(START, "detect_new_datasets")
    workflow.add_edge("detect_new_datasets", "train_on_new_datasets")
    workflow.add_edge("train_on_new_datasets", "evaluate_and_promote")
    workflow.add_edge("evaluate_and_promote", "active_learning")
    workflow.add_edge("active_learning", END)

    compiled_workflow = workflow.compile()

    print("[LangGraph] Starting MLOps Automation Pipeline...")

    try:
        while True:
            state = compiled_workflow.invoke(initial_state)

            for msg in state["messages"]:
                print(f"[{datetime.now().isoformat()}] {msg}")

            state["messages"] = []

            initial_state = state

            time.sleep(600)

    except KeyboardInterrupt:
        print("\n[LangGraph] Gracefully shutting down MLOps automation pipeline...")
    except Exception as e:
        print(f"\n[LangGraph] Error in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
