import psycopg2
import os

# PostgreSQL 데이터베이스 설정 정보
hostname = 'localhost'  # 로컬 호스트에서 실행할 경우
database = 'postgres'
username = 'postgres'
password = '0000'
port_id = 5432

# 지정된 폴더 내의 모든 파일 처리
folder_path = "C:\\testtest"
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    # 텍스트 파일인 경우에만 처리
    if os.path.isfile(file_path) and file_path.endswith(".txt"):
        with open(file_path, 'r') as file:
            data = file.readline().strip()
            values = data.split(', ')

            conn = psycopg2.connect(
                host=hostname,
                dbname=database,
                user=username,
                password=password,
                port=port_id
            )
            cur = conn.cursor()
            query = '''
                INSERT INTO cable_test (diameter, length, product_type, plant, result) VALUES (%s, %s, %s, %s, %s)
            '''
            try:
                cur.execute(query, values)
                conn.commit()  # 변경사항 저장
                file.close()  # 파일 작업을 마치고 파일을 닫습니다.
                os.remove(file_path)  # 파일 삭제
                print(f"Data from {filename} inserted and the file was deleted successfully.")
            except Exception as e:
                print(f"An error occurred while inserting data from {filename}: {e}")
            finally:
                cur.close()
                conn.close()
print("데이터 INSERT 완료 ...")