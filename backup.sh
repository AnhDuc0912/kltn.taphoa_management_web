#!/bin/bash
# Backup uploads và database từ container ra host
WEB_CONTAINER=flask_pgvector_shop_web
BACKUP_DIR=./backup

# Đọc biến môi trường từ file .env
source .env

sudo mkdir -p $BACKUP_DIR

# 1. Backup uploads folder từ container ra host
# (uploads phải nằm trong /app/uploads trong container)
sudo docker cp $WEB_CONTAINER:/app/uploads/ $BACKUP_DIR/

echo "Backup xong! Kiểm tra thư mục $BACKUP_DIR để lấy file về local."