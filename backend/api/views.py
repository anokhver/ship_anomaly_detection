# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from pathlib import Path
import subprocess


@csrf_exempt
def gettrips(request):
    trips_file = Path(__file__).parent / "visualization/clean_data/trips.json"
    with open(trips_file, "r") as f:
        trips = json.load(f)
    return JsonResponse(trips, safe=False)


@csrf_exempt
def eval(request):
    if request.method != "GET":
        return JsonResponse({"error": "Only GET allowed"}, status=405)

    model = request.GET.get("model")
    trip = request.GET.get("trip")
    if not model or not trip:
        return JsonResponse({"error": "Missing model or trip parameter"}, status=400)

    # Call the data visualizer script with subprocess
    script_path = Path(__file__).parent / "visualization/data_visualizer.py"
    visualization_dir = Path(__file__).parent / "visualization"
    result = subprocess.run(
        ["python3", str(script_path), "-d", model, trip],
        capture_output=True,
        text=True,
        cwd=visualization_dir  # <-- set working directory
    )

    if result.returncode != 0:
        return JsonResponse({
            "error": "Failed to evaluate trip",
            "stderr": result.stderr,
            "stdout": result.stdout
        }, status=500)

    trip_file = Path(__file__).parent / "visualization/trip.json"
    if not trip_file.exists():
        return JsonResponse({"error": "Trip data not found"}, status=404)
    with open(trip_file, "r") as f:
        trip_data = json.load(f)
    # Ensure trip_data is a list
    if isinstance(trip_data, dict) and "points" in trip_data:
        trip_data = trip_data["points"]
    return JsonResponse(trip_data, safe=False)
    


