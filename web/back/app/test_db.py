import pymysql

try:
    conn = pymysql.connect(
        host='192.168.56.101',
        user='plant_user',
        password='QWERTY@',
        database='plant_db',
        port=3306
    )
    print("✅ Подключение к БД успешно!")
    conn.close()
except Exception as e:
    print(f"❌ Ошибка: {e}")