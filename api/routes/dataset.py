from flask import Blueprint, request, jsonify
from api.services.dataset_service import DatasetService

dataset_bp = Blueprint("dataset", __name__)
dataset_service = DatasetService()


@dataset_bp.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        s3_path = dataset_service.upload_dataset(file)
        if s3_path:
            return jsonify(
                {"message": "Dataset uploaded successfully", "s3_path": s3_path}
            ), 200
        else:
            return jsonify({"error": "Failed to upload to S3"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dataset_bp.route("", methods=["GET"])
def list_datasets():
    try:
        datasets = dataset_service.list_datasets()
        return jsonify(datasets), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@dataset_bp.route("/<dataset_id>", methods=["GET"])
def get_dataset(dataset_id):
    try:
        dataset_info = dataset_service.get_dataset(dataset_id)
        return jsonify(dataset_info), 200
    except dataset_service.aws.s3.exceptions.NoSuchKey:
        return jsonify({"error": "Dataset not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
