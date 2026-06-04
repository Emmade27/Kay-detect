document.addEventListener("DOMContentLoaded", () => {
  const imageInput = document.getElementById("imageInput");
  const previewContainer = document.getElementById("previewContainer");
  const clearPreviewBtn = document.getElementById("clearPreviewBtn");
  const predictionForm = document.getElementById("predictionForm");
  const loadingState = document.getElementById("loadingState");
  const cameraDataInput = document.getElementById("cameraDataInput");
  const startCameraBtn = document.getElementById("startCameraBtn");
  const captureBtn = document.getElementById("captureBtn");
  const retakeBtn = document.getElementById("retakeBtn");
  const cameraVideo = document.getElementById("cameraVideo");
  const cameraCanvas = document.getElementById("cameraCanvas");
  const cameraHelpText = document.getElementById("cameraHelpText");

  let currentStream = null;

  function setPreview(src) {
    if (!previewContainer) return;
    previewContainer.innerHTML = "";
    const img = document.createElement("img");
    img.src = src;
    img.alt = "Selected preview";
    previewContainer.appendChild(img);
  }

  function resetPreview() {
    if (!previewContainer) return;
    previewContainer.innerHTML = "No image selected yet.";
    if (imageInput) imageInput.value = "";
    if (cameraDataInput) cameraDataInput.value = "";
  }

  if (imageInput) {
    imageInput.addEventListener("change", (event) => {
      const file = event.target.files?.[0];
      if (!file) {
        resetPreview();
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
        if (cameraDataInput) cameraDataInput.value = "";
      };
      reader.readAsDataURL(file);
    });
  }

  if (clearPreviewBtn) {
    clearPreviewBtn.addEventListener("click", () => resetPreview());
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (cameraHelpText) cameraHelpText.textContent = "Camera is not supported in this browser.";
      return;
    }
    try {
      currentStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      cameraVideo.srcObject = currentStream;
      if (cameraHelpText) cameraHelpText.textContent = "Camera started. Position the tomato and capture the image.";
    } catch (error) {
      if (cameraHelpText) cameraHelpText.textContent = "Camera access denied or unavailable.";
    }
  }

  function stopCamera() {
    if (currentStream) {
      currentStream.getTracks().forEach(track => track.stop());
      currentStream = null;
    }
  }

  if (startCameraBtn) startCameraBtn.addEventListener("click", startCamera);

  if (captureBtn) {
    captureBtn.addEventListener("click", () => {
      if (!cameraVideo || !cameraCanvas) return;
      if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
        if (cameraHelpText) cameraHelpText.textContent = "Start the camera before capturing.";
        return;
      }
      cameraCanvas.width = cameraVideo.videoWidth;
      cameraCanvas.height = cameraVideo.videoHeight;
      const ctx = cameraCanvas.getContext("2d");
      ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
      const dataUrl = cameraCanvas.toDataURL("image/png");
      if (cameraDataInput) cameraDataInput.value = dataUrl;
      if (imageInput) imageInput.value = "";
      setPreview(dataUrl);
      if (cameraHelpText) cameraHelpText.textContent = "Image captured. You can predict now or retake.";
    });
  }

  if (retakeBtn) {
    retakeBtn.addEventListener("click", () => {
      if (cameraDataInput) cameraDataInput.value = "";
      resetPreview();
      startCamera();
    });
  }

  if (predictionForm) {
    predictionForm.addEventListener("submit", () => {
      if (loadingState) loadingState.classList.remove("d-none");
    });
  }

  window.addEventListener("beforeunload", stopCamera);
});
