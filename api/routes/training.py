from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from api.services.training_service import TrainingService

training_bp = Blueprint("training", __name__)
training_service = TrainingService()


@training_bp.route("/start", methods=["POST"])
@jwt_required()
def start_training():
    data = request.get_json()

    required_fields = ["dataset_id", "hyperparameters"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        job_id = training_service.start_training(
            dataset_id=data["dataset_id"], hyperparameters=data["hyperparameters"]
        )
        return jsonify(
            {"message": "Training job started successfully", "job_id": job_id}
        ), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/jobs", methods=["GET"])
@jwt_required()
def list_jobs():
    try:
        jobs = training_service.list_jobs()
        return jsonify(jobs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/jobs/<job_id>", methods=["GET"])
@jwt_required()
def get_job_status(job_id):
    try:
        job = training_service.get_job_status(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/jobs/<job_id>/metrics", methods=["GET"])
@jwt_required()
def get_training_metrics(job_id):
    try:
        metrics = training_service.get_training_metrics(job_id)
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
