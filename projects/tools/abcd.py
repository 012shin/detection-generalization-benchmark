import os
import xml.etree.ElementTree as ET

# XML 파일이 저장된 디렉터리 경로
directory = '/home/dataset/detectron2/comic/Annotations/'

# 디렉터리 내 모든 파일 처리
for filename in os.listdir(directory):
    # 파일 확장자가 .xml인 경우만 처리
    if filename.endswith('.xml'):
        filepath = os.path.join(directory, filename)
        try:
            # XML 파일 열기 및 내용 읽기
            tree = ET.parse(filepath)
            root = tree.getroot()
            print(f"파일 이름: {filename}")
            print("내용:")
            # XML 내용을 보기 좋게 출력
            for elem in root.iter():
                print(f"{elem.tag}: {elem.text}")
            print("-" * 50)  # 구분선
        except Exception as e:
            print(f"파일 {filename} 처리 중 오류 발생: {e}")