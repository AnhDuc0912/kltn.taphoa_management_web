# app/routes/uploads.py
import os
from flask import Blueprint, send_from_directory, current_app

bp = Blueprint("uploads", __name__)

@bp.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(current_app.config["UPLOAD_DIR"], filename)

