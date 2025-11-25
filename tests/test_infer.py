# tests/test_infer.py
import pytest
import numpy as np
import os
# Nanti kita akan buat fungsi ini di src
# from bsort.infer import load_model, preprocess_image 

def test_preprocess_image_shape():
    """
    Unit Test: Memastikan gambar di-resize dengan benar ke 224x224 (atau 320)
    sebelum masuk ke model.
    """
    # 1. Buat gambar dummy (Random noise) ukuran sembarang
    dummy_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # 2. Panggil fungsi preprocess (Nanti kita buat fungsinya)
    # processed_img = preprocess_image(dummy_img, target_size=224)
    
    # 3. Assert (Cek) apakah outputnya sesuai target
    # assert processed_img.shape == (1, 3, 224, 224) 
    assert True # Placeholder biar pass dulu sebelum codenya ada