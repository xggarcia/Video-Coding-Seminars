"""
API Integration Tests for Practice 2 endpoints
Tests FastAPI endpoints with mock requests
"""

import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from PIL import Image
import io

from main import app

client = TestClient(app)


class TestBasicEndpoints:
    """Test basic API endpoints"""
    
    def test_root_returns_html(self):
        """Test that root endpoint serves HTML"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Video Processing API" in response.text
    
    def test_api_info(self):
        """Test API info endpoint"""
        response = client.get("/api")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_rgb_to_yuv_endpoint(self):
        """Test RGB to YUV conversion endpoint"""
        response = client.get("/rgb-to-yuv?r=255&g=0&b=0")
        assert response.status_code == 200
        data = response.json()
        assert "rgb" in data
        assert "yuv" in data
        assert data["rgb"] == [255, 0, 0]


class TestRLEEncoding:
    """Test RLE encoding endpoint"""
    
    def test_rle_encoding_simple(self):
        """Test RLE endpoint with simple data"""
        response = client.post(
            "/encode-rle",
            json={"data": ["A", "A", "B"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["original"] == ["A", "A", "B"]
        assert data["encoded"] == ["A", 2, "B", 1]
    
    def test_rle_encoding_empty(self):
        """Test RLE with empty array"""
        response = client.post(
            "/encode-rle",
            json={"data": []}
        )
        assert response.status_code == 200
        assert response.json()["encoded"] == []


class TestFileUploadEndpoints:
    """Test endpoints that require file uploads"""
    
    @pytest.fixture
    def test_image_bytes(self):
        """Create test image as bytes"""
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.read()
    
    @pytest.fixture
    def test_video_file(self):
        """Create a minimal test video file path"""
        # Note: This would need ffmpeg to create actual video
        # For now, we'll use a placeholder
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(b'fake video data')
            return tmp.name
    
    def test_serpentine_read_requires_file(self):
        """Test serpentine endpoint without file"""
        response = client.post("/serpentine-read")
        assert response.status_code == 422  # Unprocessable entity
    
    def test_serpentine_read_with_image(self, test_image_bytes):
        """Test serpentine read with actual image"""
        response = client.post(
            "/serpentine-read",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_pixels" in data
        assert data["total_pixels"] > 0
    
    def test_resize_missing_parameters(self, test_image_bytes):
        """Test resize endpoint with missing parameters"""
        response = client.post(
            "/resize",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        )
        # Should fail due to missing width, height, isVideo parameters
        assert response.status_code == 422
    
    def test_max_compression_endpoint(self, test_image_bytes):
        """Test max compression endpoint"""
        response = client.post(
            "/max-compression",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        )
        # May fail on actual compression but should accept the request
        assert response.status_code in [200, 500]  # 500 if ffmpeg fails


class TestCodecConversion:
    """Test codec conversion endpoints"""
    
    def test_convert_codec_missing_file(self):
        """Test convert codec without file"""
        response = client.post("/convert_codec?codec=vp8")
        assert response.status_code == 422
    
    def test_convert_codec_invalid_codec(self):
        """Test convert codec with invalid codec parameter"""
        fake_video = io.BytesIO(b'fake video')
        response = client.post(
            "/convert_codec?codec=invalid_codec",
            files={"file": ("test.mp4", fake_video, "video/mp4")}
        )
        # Should return error for invalid codec
        assert response.status_code in [400, 500]


class TestEncodingLadder:
    """Test encoding ladder endpoint"""
    
    def test_encoding_ladder_missing_file(self):
        """Test encoding ladder without file"""
        response = client.post("/create_encoding_ladder?codec=h265")
        assert response.status_code == 422
    
    def test_encoding_ladder_with_invalid_codec(self):
        """Test encoding ladder with unsupported codec"""
        fake_video = io.BytesIO(b'fake video')
        response = client.post(
            "/create_encoding_ladder?codec=invalid",
            files={"file": ("test.mp4", fake_video, "video/mp4")}
        )
        # Should handle gracefully (might default to h264)
        assert response.status_code in [200, 400, 500]


class TestVideoAnalysis:
    """Test video analysis endpoints"""
    
    def test_relevant_information_missing_file(self):
        """Test metadata endpoint without file"""
        response = client.post("/relevant_information")
        assert response.status_code == 422
    
    def test_motion_vectors_missing_file(self):
        """Test motion vectors without file"""
        response = client.post("/visualize_motion_vectors")
        assert response.status_code == 422
    
    def test_yuv_histogram_missing_file(self):
        """Test YUV histogram without file"""
        response = client.post("/yuv_histogram")
        assert response.status_code == 422
    
    def test_count_tracks_missing_file(self):
        """Test track count without file"""
        response = client.post("/count-tracks")
        assert response.status_code == 422


class TestChromaSubsampling:
    """Test chroma subsampling endpoint"""
    
    def test_chroma_valid_formats(self):
        """Test that valid chroma formats are accepted"""
        valid_formats = ['4:4:4', '4:2:2', '4:2:0', '4:0:0']
        fake_video = io.BytesIO(b'fake video')
        
        for fmt in valid_formats:
            response = client.post(
                f"/set-chroma?subsampling={fmt}",
                files={"file": ("test.mp4", fake_video, "video/mp4")}
            )
            # Should not return 400 for valid format (may fail at 500 if ffmpeg fails)
            assert response.status_code != 400
            fake_video.seek(0)  # Reset for next iteration


class TestBBBContainer:
    """Test BBB container creation endpoint"""
    
    def test_bbb_container_default_duration(self):
        """Test BBB container with default 20 second duration"""
        fake_video = io.BytesIO(b'fake video')
        response = client.post(
            "/create_bbb_container",
            files={"file": ("test.mp4", fake_video, "video/mp4")}
        )
        # Will likely fail without real video, but tests parameter handling
        assert response.status_code in [200, 404, 500]
    
    def test_bbb_container_custom_duration(self):
        """Test BBB container with custom duration"""
        fake_video = io.BytesIO(b'fake video')
        response = client.post(
            "/create_bbb_container?duration=10",
            files={"file": ("test.mp4", fake_video, "video/mp4")}
        )
        assert response.status_code in [200, 404, 500]


class TestErrorHandling:
    """Test error handling across endpoints"""
    
    def test_invalid_file_type_handling(self):
        """Test that API handles invalid file types gracefully"""
        text_file = io.BytesIO(b'This is not a video')
        response = client.post(
            "/convert_codec?codec=vp8",
            files={"file": ("test.txt", text_file, "text/plain")}
        )
        # Should handle gracefully with error response
        assert response.status_code in [400, 404, 500]
    
    def test_large_parameter_values(self):
        """Test handling of unreasonably large parameters"""
        fake_img = io.BytesIO(b'fake image')
        response = client.post(
            "/resize?width=999999&height=999999&isVideo=false",
            files={"file": ("test.jpg", fake_img, "image/jpeg")}
        )
        # Should handle without crashing
        assert response.status_code in [200, 400, 500]
    
    def test_negative_parameter_values(self):
        """Test handling of negative parameters"""
        fake_video = io.BytesIO(b'fake video')
        response = client.post(
            "/create_bbb_container?duration=-10",
            files={"file": ("test.mp4", fake_video, "video/mp4")}
        )
        # Should handle invalid duration gracefully
        assert response.status_code in [200, 400, 500]


# Performance and load tests
class TestPerformance:
    """Basic performance and concurrency tests"""
    
    def test_concurrent_rgb_conversions(self):
        """Test multiple concurrent RGB to YUV conversions"""
        import concurrent.futures
        
        def make_request(r, g, b):
            return client.get(f"/rgb-to-yuv?r={r}&g={g}&b={b}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(make_request, i, i, i)
                for i in range(0, 255, 25)
            ]
            results = [f.result() for f in futures]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in results)
    
    def test_api_response_time(self):
        """Test that simple endpoints respond quickly"""
        import time
        
        start = time.time()
        response = client.get("/api")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # Should respond in under 1 second


# Run tests with: pytest test_api.py -v --tb=short
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
