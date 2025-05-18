from flask import Blueprint, request, jsonify
from api.services.assistant_service import AssistantService

assistant_bp = Blueprint("assistant", __name__)
assistant_service = AssistantService()


@assistant_bp.route("/query", methods=["POST"])
def query_assistant():
    data = request.form

    if "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    try:
        result = assistant_service.process_query(
            question=data["question"], image_file=request.files.get("image")
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
