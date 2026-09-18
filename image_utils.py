from PIL import Image
import base64
import hashlib
import io
import streamlit as st

# ── image metadata cache (populated by validate_and_process_image) ────────────
_last_image_meta: dict = {}

def get_last_image_meta() -> dict:
    """Return a copy of the metadata dict from the most recent processing call."""
    return dict(_last_image_meta)


def process_image_file(filepath: str):
    """
    Process an image from a file path (no Streamlit upload object needed).
    Used by run_tests.py.  Returns (image_data_b64, media_type, error_or_None).
    """
    import pathlib
    import io as _io
    p = pathlib.Path(filepath)
    if not p.exists():
        return None, None, f"File not found: {filepath}"
    try:
        raw = p.read_bytes()
        image = Image.open(_io.BytesIO(raw))
        processed_image, was_compressed = compress_image_for_api(image)
        image_data, media_type = image_to_base64(processed_image, was_compressed)
        global _last_image_meta
        _last_image_meta = {
            "source_dimensions_px": list(image.size),
            "sent_media_type":      media_type,
            "sent_dimensions_px":   list(processed_image.size),
            "resize_applied":       was_compressed,
            "compression_quality":  None,
        }
        return image_data, media_type, None
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"


def compress_image_for_api(image, max_size_mb=4.5):
    """
    Compress image to ensure it stays under Claude API limits after base64 encoding.
    Base64 encoding increases size by ~33%, so we target 4.5MB to stay under 5MB limit.
    Only compresses if the image exceeds the size limit.
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # First, check if the image needs compression at all
    # Try saving in original format to check size
    test_buffer = io.BytesIO()
    save_format = 'PNG' if image.mode == 'RGBA' else 'JPEG'
    
    if save_format == 'JPEG':
        if image.mode in ('RGBA', 'LA', 'P'):
            temp_image = image.convert('RGB')
        else:
            temp_image = image
        temp_image.save(test_buffer, format='JPEG', quality=95)
    else:
        image.save(test_buffer, format='PNG')
    
    # If image is already small enough, return it as-is (no compression needed)
    if len(test_buffer.getvalue()) <= max_size_bytes:
        return image, False  # False = no compression was needed
    
    # Image needs compression - proceed with compression logic
    compressed_image = image.copy()
    
    # Convert to RGB if necessary for JPEG compression
    if compressed_image.mode in ('RGBA', 'LA', 'P'):
        compressed_image = compressed_image.convert('RGB')
    
    # Initial compression attempt with high quality
    quality = 95
    
    while quality > 20:  # Don't go below 20% quality
        buffer = io.BytesIO()
        compressed_image.save(buffer, format='JPEG', quality=quality, optimize=True)
        
        buffer_size = len(buffer.getvalue())
        
        if buffer_size <= max_size_bytes:
            buffer.seek(0)
            return Image.open(buffer), True  # True = compression was applied
        
        quality -= 10
    
    # If quality reduction isn't enough, resize the image
    original_width, original_height = compressed_image.size
    scale_factor = 0.9
    
    while scale_factor > 0.3:  # Don't scale below 30% of original
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        resized_image = compressed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        resized_image.save(buffer, format='JPEG', quality=85, optimize=True)
        
        buffer_size = len(buffer.getvalue())
        
        if buffer_size <= max_size_bytes:
            buffer.seek(0)
            return Image.open(buffer), True
        
        scale_factor -= 0.1
    
    # Last resort: very aggressive compression
    final_width = int(original_width * 0.5)
    final_height = int(original_height * 0.5)
    final_image = compressed_image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    final_image.save(buffer, format='JPEG', quality=70, optimize=True)
    buffer.seek(0)
    
    return Image.open(buffer), True

def image_to_base64(image, was_compressed=False):
    """Convert PIL Image to base64 string for API, preserving format if not compressed"""
    buffer = io.BytesIO()
    
    if was_compressed:
        # Compressed images are always JPEG
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=90, optimize=True)
        media_type = 'image/jpeg'
    else:
        # Preserve original format for uncompressed images
        if image.mode == 'RGBA':
            image.save(buffer, format='PNG')
            media_type = 'image/png'
        else:
            if image.mode in ('LA', 'P'):
                image = image.convert('RGB')
            image.save(buffer, format='JPEG', quality=95)
            media_type = 'image/jpeg'
    
    image_data = base64.b64encode(buffer.getvalue()).decode()
    return image_data, media_type

def validate_and_process_image(uploaded_file):
    """Validate and process uploaded image for analysis"""
    
    if uploaded_file is None:
        return None, None, "No file uploaded"
    
    # Check file size (limit to 20MB for upload)
    if uploaded_file.size > 20 * 1024 * 1024:
        return None, None, "Image file too large. Please upload an image smaller than 20MB."
    
    # Check file type
    allowed_types = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension not in allowed_types:
        return None, None, f"Unsupported file type. Please upload: {', '.join(allowed_types)}"
    
    try:
        # Open image
        image = Image.open(uploaded_file)
        
        # Compress image only if needed for API (>4.5MB)
        processed_image, was_compressed = compress_image_for_api(image)
        
        # Convert to base64, preserving format if not compressed
        image_data, media_type = image_to_base64(processed_image, was_compressed)
        
        # Show compression info only if image was actually compressed
        if was_compressed:
            original_size_mb = uploaded_file.size / (1024 * 1024)
            estimated_compressed_size_mb = len(image_data) / (1024 * 1024)
            st.info(f"Image compressed from {original_size_mb:.1f}MB to ~{estimated_compressed_size_mb:.1f}MB for analysis")
        
        # ── populate metadata cache for logging ──────────────────────────────
        global _last_image_meta
        _last_image_meta = {
            "source_dimensions_px": list(image.size),
            "sent_media_type":      media_type,
            "sent_dimensions_px":   list(processed_image.size),
            "resize_applied":       was_compressed,
            "compression_quality":  None,  # loop internals not exposed here
        }
        return image, (image_data, media_type), None

    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"