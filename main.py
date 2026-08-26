from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Change host/port to point to central deployment when going live
SIGNALING_SERVER_HOST = "localhost:8000"

@app.get("/", response_class=HTMLResponse)
async def get_student_kiosk(request: Request):
    return templates.TemplateResponse(
        "student.html",
        {"request": request, "signaling_host": SIGNALING_SERVER_HOST}
    )