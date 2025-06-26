"""Entry point for the YOLO detection system"""



if __name__ == "__main__":
    try:
        system = DetectionSystem()
        system.run()
    except Exception as e:
        print(f"Error starting detection system: {e}")
        import traceback
        traceback.print_exc()