import psycopg2
import pandas as pd
import os

# Thông tin kết nối từ .env (thay bằng giá trị thực nếu cần)
DB_HOST = 'ducdatphat.id.vn'
DB_PORT = 5001
DB_NAME = 'hango'
DB_USER = 'admin'
DB_PASS = 'Duc091203@'

try:
    # Kết nối CSDL
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    
    # Query bảng sku_captions
    query_captions = "SELECT * FROM sku_captions WHERE model_name = 'gemini-2.0-flash-exp'; "
    df_captions = pd.read_sql_query(query_captions, conn)
    
    # Query bảng sku_images (nếu cần kết hợp)
    query_images = "SELECT si.* FROM sku_images si JOIN skus s ON (si.sku_id = s.id)"
    df_images = pd.read_sql_query(query_images, conn)
    
    # Đóng kết nối
    conn.close()
    
    # Lưu thành CSV
    df_captions.to_csv('sku_captions_export.csv', index=False, sep=';')  # Sử dụng ; làm delimiter như bảng gốc
    df_images.to_csv('sku_images_export.csv', index=False, sep=';')
    
    print("Đã xuất dữ liệu thành công vào sku_captions_export.csv và sku_images_export.csv")
except Exception as e:
    print(f"Lỗi khi kết nối hoặc xuất dữ liệu: {e}")