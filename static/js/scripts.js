function startLive() {
    document.getElementById('liveFeed').src = "/start_live";
}

function stopLive() {
    document.getElementById('liveFeed').src = "";
    fetch("/stop_live");
}

function stopVideoProcessing() {
    fetch("/stop_video_processing", { method: "POST" })
        .then(response => {
            // Optionally handle response if needed
            console.log("Stopped video processing");
        });
}
