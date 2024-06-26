from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers['content-type'] == 'text/html; charset=utf-8'


def test_upload_image():
    test_image_path = "test_data/cars.jpg"
    with open(test_image_path, "rb") as image:
        files = {'file_upload': ('cars.jpg', image, 'image/jpeg')}
        data = {'confidence': '0.5'}
        response = client.post("/upload_image/", files=files, data=data)

        if response.status_code != 200:
            print(response.json())

        assert response.status_code == 200
        assert 'base64_image' in response.json()


def test_upload_video():
    test_video_path = "test_data/test_video.mp4"
    with open(test_video_path, "rb") as video:
        files = {'file_upload': ('test_video.mp4', video, 'video/mp4')}
        response = client.post("/upload_video/", files=files)
        assert response.status_code == 200
        assert 'video_id' in response.json()


def test_process_video_not_found():
    response = client.get("/process_video/?video_id=nonexistent.mp4")
    assert response.status_code == 404
    assert response.json() == {"detail": "Video not found"}


def test_process_video():
    with patch('your_application_file.get_model') as mock_get_model:
        net_mock = MagicMock()
        classes_mock = ['car', 'person']  # Example classes
        mock_get_model.return_value = (net_mock, classes_mock)

        video_id = "valid_video.mp4"
        response = client.get(f"/process_video/?video_id={video_id}&confidence=0.5")
        assert response.status_code == 200
        assert response.headers['content-type'].startswith("multipart/x-mixed-replace")


def test_upload_image_no_file():
    data = {'confidence': '0.5'}
    response = client.post("/upload_image/", data=data)
    assert response.status_code == 422


def test_upload_video_no_file():
    response = client.post("/upload_video/")
    assert response.status_code == 422
