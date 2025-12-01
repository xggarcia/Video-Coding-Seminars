"""
Unit tests for p2_logic module
Tests cover ColorTranslator, DataSerializer, FFmpegAuto, DCT_Converter, and DWT_Converter
"""

import pytest
import subprocess
import os
import tempfile
import numpy as np
from PIL import Image
from p2_logic import ColorTranslator, DataSerializer, FFmpegAuto, DCT_Converter, DWT_Converter


class TestColorTranslator:
    """Test RGB to YUV color space conversions"""
    
    def test_rgb_to_yuv_black(self):
        """Test conversion of pure black"""
        Y, U, V = ColorTranslator.rgb_to_yuv(0, 0, 0)
        assert Y == 0.0
        assert U == 0.0
        assert V == 0.0
    
    def test_rgb_to_yuv_white(self):
        """Test conversion of pure white"""
        Y, U, V = ColorTranslator.rgb_to_yuv(255, 255, 255)
        assert Y == 255.0
        assert abs(U) < 0.01  # Should be close to 0
        assert abs(V) < 0.01  # Should be close to 0
    
    def test_rgb_to_yuv_red(self):
        """Test conversion of pure red"""
        Y, U, V = ColorTranslator.rgb_to_yuv(255, 0, 0)
        assert 70 < Y < 80  # Approximate luma for red
        assert V > 100  # Red has high V component
    
    def test_yuv_to_rgb_black(self):
        """Test reverse conversion of black"""
        R, G, B = ColorTranslator.rgb_to_yuv(0, 0, 0, mode='YUV_to_RGB')
        assert R == 0
        assert G == 0
        assert B == 0
    
    def test_yuv_to_rgb_white(self):
        """Test reverse conversion of white"""
        R, G, B = ColorTranslator.rgb_to_yuv(255, 0, 0, mode='YUV_to_RGB')
        assert R == 255
        assert G == 255
        assert B == 255
    
    def test_rgb_yuv_roundtrip(self):
        """Test that RGB -> YUV -> RGB preserves values (approximately)"""
        original = (128, 64, 192)
        Y, U, V = ColorTranslator.rgb_to_yuv(*original)
        R, G, B = ColorTranslator.rgb_to_yuv(int(Y), int(U), int(V), mode='YUV_to_RGB')
        
        # Allow small rounding errors
        assert abs(R - original[0]) < 5
        assert abs(G - original[1]) < 5
        assert abs(B - original[2]) < 5


class TestDataSerializer:
    """Test data serialization and encoding methods"""
    
    def test_run_length_encoding_simple(self):
        """Test RLE on simple sequence"""
        input_data = ['A', 'A', 'B']
        result = DataSerializer.run_length_encoding(input_data)
        assert result == ['A', 2, 'B', 1]
    
    def test_run_length_encoding_all_same(self):
        """Test RLE on sequence with all same elements"""
        input_data = ['X', 'X', 'X', 'X']
        result = DataSerializer.run_length_encoding(input_data)
        assert result == ['X', 4]
    
    def test_run_length_encoding_all_different(self):
        """Test RLE on sequence with all different elements"""
        input_data = ['A', 'B', 'C', 'D']
        result = DataSerializer.run_length_encoding(input_data)
        assert result == ['A', 1, 'B', 1, 'C', 1, 'D', 1]
    
    def test_run_length_encoding_empty(self):
        """Test RLE on empty array"""
        result = DataSerializer.run_length_encoding([])
        assert result == []
    
    def test_run_length_encoding_numbers(self):
        """Test RLE with numeric values"""
        input_data = [1, 1, 2, 2, 2, 3]
        result = DataSerializer.run_length_encoding(input_data)
        assert result == [1, 2, 2, 3, 3, 1]
    
    def test_serpentine_read_requires_file(self):
        """Test that serpentine_read raises error for missing file"""
        with pytest.raises(FileNotFoundError):
            DataSerializer.serpentine_read("nonexistent_file.jpg")
    
    def test_serpentine_read_with_temp_image(self):
        """Test serpentine read on a temporary test image"""
        # Create a simple 3x3 test image
        img = Image.new('RGB', (3, 3), color='white')
        pixels = img.load()
        
        # Set some pixels to black for pattern
        pixels[0, 0] = (0, 0, 0)
        pixels[1, 1] = (128, 128, 128)
        pixels[2, 2] = (255, 0, 0)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            result = DataSerializer.serpentine_read(tmp_path)
            # Should return 9 pixels (3x3 image)
            assert len(result) == 9
            # First pixel should be black
            assert result[0] == (0, 0, 0)
        finally:
            os.unlink(tmp_path)
    
    def test_important_information_missing_file(self):
        """Test metadata extraction with missing file"""
        with pytest.raises(FileNotFoundError):
            DataSerializer.important_information("nonexistent_video.mp4")


class TestFFmpegAuto:
    """Test FFmpeg wrapper methods"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def test_image(self, temp_dir):
        """Create a test image file"""
        img = Image.new('RGB', (100, 100), color='blue')
        path = os.path.join(temp_dir, 'test.jpg')
        img.save(path)
        return path
    
    def test_resize_validates_dimensions(self):
        """Test that resize validates dimension inputs"""
        # This should not crash with valid dimensions
        # Note: Will fail on actual execution without valid input file
        # but validates the parameter handling
        try:
            FFmpegAuto.resize("dummy.mp4", 1920, 1080, "output.mp4")
        except subprocess.CalledProcessError:
            pass  # Expected - file doesn't exist
        except TypeError:
            pytest.fail("Resize should accept integer dimensions")
    
    def test_max_compression_creates_temp_file(self, temp_dir, test_image):
        """Test that max_compression properly manages temp files"""
        output = os.path.join(temp_dir, 'compressed.jpg')
        
        # This will likely fail due to ffmpeg quantization requirements
        # but tests the file management logic
        try:
            FFmpegAuto.max_compression(test_image, output)
        except subprocess.CalledProcessError:
            # Expected - may fail on actual encoding
            pass
        
        # Check that temp file is cleaned up
        temp_files = [f for f in os.listdir(temp_dir) if 'temp_resized' in f]
        assert len(temp_files) == 0, "Temporary files should be cleaned up"
    
    def test_set_chroma_subsampling_invalid_format(self):
        """Test that invalid subsampling format raises ValueError"""
        with pytest.raises(ValueError, match="Unsupported subsampling"):
            FFmpegAuto.set_chroma_subsampling("test.mp4", "output.mp4", "9:9:9")
    
    def test_set_chroma_subsampling_valid_formats(self):
        """Test that all valid subsampling formats are accepted"""
        valid_formats = ['4:4:4', '4:2:2', '4:2:0', '4:0:0']
        
        for fmt in valid_formats:
            try:
                # Will fail on execution but should pass validation
                FFmpegAuto.set_chroma_subsampling("test.mp4", "output.mp4", fmt)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass  # Expected - file doesn't exist
            except ValueError:
                pytest.fail(f"Format {fmt} should be valid")


class TestDCTConverter:
    """Test Discrete Cosine Transform operations"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def test_grayscale_image(self, temp_dir):
        """Create a test grayscale image"""
        # Create 64x64 grayscale image for DCT (divisible by 8)
        img_array = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        path = os.path.join(temp_dir, 'test_gray.png')
        img.save(path)
        return path
    
    def test_dct_converter_initialization(self):
        """Test DCT converter can be initialized with custom block size"""
        converter = DCT_Converter(block_size=16)
        assert converter.block_size == 16
    
    def test_dct_converter_default_block_size(self):
        """Test DCT converter uses default 8x8 block size"""
        converter = DCT_Converter()
        assert converter.block_size == 8
    
    def test_apply_dct_missing_file(self):
        """Test that apply_dct raises error for missing file"""
        converter = DCT_Converter()
        with pytest.raises(FileNotFoundError):
            converter.apply_dct("nonexistent.jpg", "dct.jpg", "idct.jpg")
    
    def test_apply_dct_creates_output_files(self, temp_dir, test_grayscale_image):
        """Test that DCT creates both output files"""
        converter = DCT_Converter()
        dct_out = os.path.join(temp_dir, 'dct_output.jpg')
        idct_out = os.path.join(temp_dir, 'idct_output.jpg')
        
        converter.apply_dct(test_grayscale_image, dct_out, idct_out)
        
        assert os.path.exists(dct_out), "DCT visualization should be created"
        assert os.path.exists(idct_out), "IDCT reconstruction should be created"


class TestDWTConverter:
    """Test Discrete Wavelet Transform operations"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def test_grayscale_image(self, temp_dir):
        """Create a test grayscale image with even dimensions"""
        # Create 64x64 grayscale image (even dimensions for DWT)
        img_array = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        path = os.path.join(temp_dir, 'test_gray.png')
        img.save(path)
        return path
    
    def test_dwt_converter_initialization(self):
        """Test DWT converter with different wavelets"""
        converter = DWT_Converter(wavelet='db4')
        assert converter.wavelet == 'db4'
    
    def test_dwt_converter_default_wavelet(self):
        """Test DWT converter uses Haar wavelet by default"""
        converter = DWT_Converter()
        assert converter.wavelet == 'haar'
    
    def test_apply_dwt_missing_file(self):
        """Test that apply_dwt raises error for missing file"""
        converter = DWT_Converter()
        with pytest.raises(FileNotFoundError):
            converter.apply_dwt("nonexistent.jpg", "dwt.jpg", "idwt.jpg")
    
    def test_apply_dwt_creates_output_files(self, temp_dir, test_grayscale_image):
        """Test that DWT creates both output files"""
        converter = DWT_Converter()
        dwt_out = os.path.join(temp_dir, 'dwt_output.jpg')
        idwt_out = os.path.join(temp_dir, 'idwt_output.jpg')
        
        converter.apply_dwt(test_grayscale_image, dwt_out, idwt_out)
        
        assert os.path.exists(dwt_out), "DWT visualization should be created"
        assert os.path.exists(idwt_out), "IDWT reconstruction should be created"


class TestIntegration:
    """Integration tests for complex workflows"""
    
    def test_color_conversion_preserves_brightness(self):
        """Test that RGB to YUV preserves relative brightness"""
        colors = [
            (0, 0, 0),      # Black
            (128, 128, 128), # Gray
            (255, 255, 255)  # White
        ]
        
        luminances = []
        for r, g, b in colors:
            Y, _, _ = ColorTranslator.rgb_to_yuv(r, g, b)
            luminances.append(Y)
        
        # Luminance should increase monotonically
        assert luminances[0] < luminances[1] < luminances[2]
    
    def test_rle_encoding_reduces_size_for_repetitive_data(self):
        """Test that RLE reduces size for repetitive sequences"""
        # Highly repetitive data
        repetitive = ['A'] * 100
        encoded = DataSerializer.run_length_encoding(repetitive)
        assert len(encoded) == 2  # ['A', 100]
        assert len(encoded) < len(repetitive)
    
    def test_rle_encoding_increases_size_for_unique_data(self):
        """Test that RLE increases size for non-repetitive sequences"""
        # All unique data
        unique = list(range(50))
        encoded = DataSerializer.run_length_encoding(unique)
        # Each item becomes (value, 1), so double the size
        assert len(encoded) == len(unique) * 2


# Run tests with: pytest test_p2_logic.py -v
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
