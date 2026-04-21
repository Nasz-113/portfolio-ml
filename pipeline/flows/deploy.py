from datetime import timedelta
from ml_orchestration import main

if __name__ == "__main__":
    
    main()

    main.serve(
        name="hourly_schedule",
        interval=timedelta(hours=1)
    )