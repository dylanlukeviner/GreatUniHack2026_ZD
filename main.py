import sys
import multiprocessing
from PyQt6.QtWidgets import QApplication
from gui import TutorApp

if __name__ == '__main__':
    # MANDATORY for PyTorch/WhisperX/Demucs on Windows to prevent crash loops
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    import traceback
    try:
        app = QApplication(sys.argv)
        window = TutorApp()
        window.show()
        sys.exit(app.exec())
    except Exception as run_error:
        print("--- RUNTIME ERROR BLOCK EXECUTED ---")
        traceback.print_exc()
        input("\nPress Enter to exit terminal...")