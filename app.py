# app.py
import os
from flask import Flask
from search_routes import bp as search_bp  # náº¿u báº¡n Ä‘ang dÃ¹ng
from werkzeug.routing import BuildError

# --- App & cáº¥u hÃ¬nh chung ---
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

UPLOAD_DIR = os.path.abspath(os.getenv("UPLOAD_FOLDER", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_DIR"] = UPLOAD_DIR

# --- ÄÄƒng kÃ½ cÃ¡c blueprint (routes Ä‘Ã£ tÃ¡ch) ---
from routes.dashboard import bp as dashboard_bp
from routes.uploads import bp as uploads_bp
from routes.brands import bp as brands_bp
from routes.categories import bp as categories_bp
from routes.skus import bp as skus_bp
from routes.sku_texts import bp as sku_texts_bp
from routes.sku_images import bp as sku_images_bp
from routes.import_csv import bp as import_bp
from routes.captions import bp as captions_bp
from routes.admin import bp as admin_bp
from routes import search_test

app.register_blueprint(search_bp)      # blueprint hiá»‡n cÃ³
app.register_blueprint(dashboard_bp)
app.register_blueprint(uploads_bp)
app.register_blueprint(brands_bp)
app.register_blueprint(categories_bp)
app.register_blueprint(skus_bp)
app.register_blueprint(sku_texts_bp)
app.register_blueprint(sku_images_bp)
app.register_blueprint(import_bp)
app.register_blueprint(captions_bp)
# Đảm bảo chỉ import và register admin blueprint một lần
from routes import admin
app.register_blueprint(admin.bp)  # chỉ có 1 dòng này, không duplicate
app.register_blueprint(search_test.bp)

# factory compat (náº¿u cáº§n)
def create_app():
    return app

def install_endpoint_aliases(app):
    existing = set(app.view_functions.keys())  # ví dụ: {'sku_images_bp.sku_images', ...}
    to_add = []  # (rule.rule, alias_endpoint, view_func, methods)

    # duyệt toàn bộ rule hiện có
    for rule in list(app.url_map.iter_rules()):
        ep = rule.endpoint  # ví dụ 'sku_images_bp.sku_images'
        if "." not in ep:
            continue
        alias = ep.split(".", 1)[1]  # 'sku_images'
        # nếu alias chưa tồn tại và không đụng hàng thì thêm alias
        if alias not in existing:
            view_func = app.view_functions[ep]
            methods = sorted(m for m in (rule.methods or set()) if m not in {"HEAD", "OPTIONS"})
            if not methods:
                methods = ["GET"]
            to_add.append((rule.rule, alias, view_func, methods))
            existing.add(alias)

    for rule_str, alias, view_func, methods in to_add:
        try:
            app.add_url_rule(rule_str, endpoint=alias, view_func=view_func, methods=methods)
            print(f"[alias] {alias} -> {rule_str} ({methods})")
        except Exception as e:
            print(f"[alias-skip] {alias}: {e}")

# GỌI SAU khi app.register_blueprint(...) cho tất cả blueprint
install_endpoint_aliases(app)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))

