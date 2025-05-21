from flask import Blueprint, render_template

frontend_bp = Blueprint(
    "frontend", __name__, template_folder="../../templates", static_folder="../../static"
)


@frontend_bp.route("/")
def index():
    return render_template("index.html")
