function startLiveUpdate() {
    var progressBar = document.getElementById("progressBar");
    var value = 0;
    var increment = 1;
    var intervalTime = 30; // Update every 30 milliseconds
    var totalTime = 3000; // Total time in milliseconds

    var numSteps = totalTime / intervalTime; // Number of steps needed
    var incrementSize = 100 / numSteps; // Increment value for each step

    setInterval(function () {
        if (value >= 100) {
            fetch('/pridiction').then(function (response) {
                return response.json();
            }).then(function (data) {
                document.getElementById('viewPridiction').innerHTML = data.label;
                new_char = data.label;
                if (new_char != "") {
                    text = String(document.getElementById('outputText').value)
                    last_char = String(text.charAt(text.length - 1))
                    text = text + new_char
                    document.getElementById('outputText').value = text;
                }
            }).catch(function (error) {
                console.log(error);
            });
            value = 0; // Reset value to 0 after reaching 100%
        } else {
            value += incrementSize;
        }
        progressBar.value = value;
    }, increment * intervalTime);

}

const btn = document.getElementById('btn_open_close');
const video_camera_container = document.getElementById('video_camera_container');
const progressBar = document.getElementById("progressBar");
const img_or_frames = document.getElementById('img_or_frames');
const no_video_icon = document.getElementById('no_video_icon');


window.onload = function () {

    if (sessionStorage.getItem("cameraState") == "true") {
        video_camera_container.removeChild(no_video_icon);
        startLiveUpdate();
        video_camera_container.appendChild(progressBar);
        img_or_frames.setAttribute("src", "/video");
        btn.textContent = "CLOSE CAMERA";
    }
    else{
        video_camera_container.removeChild(progressBar);
    }
}



btn.onclick = function () {
    if (btn.textContent == "OPEN CAMERA") {
        sessionStorage.setItem("cameraState", true);
        window.location.reload();
    }
    else {
        sessionStorage.setItem("cameraState", false);
        window.location.href = "/close"
        btn.textContent = "OPEN CAMERA"
    }
}

const input_text = document.getElementById('outputText');
const speak_btn = document.getElementById('speak_btn');
const clear_btn = document.getElementById('btn_clear');
const download_btn = document.getElementById('btn_download');
download_btn.style.cursor = "not-allowed";


speak_btn.onclick = function(){
    if(input_text.value != "")
    {
        var message = new SpeechSynthesisUtterance();
        message.text = input_text.value;
        window.speechSynthesis.speak(message);
        download_btn.disabled = false;
        download_btn.style.cursor = "pointer";
        download_btn.classList.add('donwload_btn_hover');
    }
}

clear_btn.onclick = function()
{
    input_text.value = "";
}

download_btn.onclick = function()
{
    if(input_text.value == "")
    {
        download_btn.disabled = true;
        download_btn.style.cursor = "not-allowed";
        download_btn.classList.remove('donwload_btn_hover');
    }
    else
    {
        fetch("/download", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({"text": input_text.value})
        }).then(response => response.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'audio.mp3';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        });
    }
}

function downloadAsMp3() {

    // Create a new instance of SpeechSynthesisUtterance
    var utterance = new SpeechSynthesisUtterance();
    utterance.text = String(document.getElementById('outputText').value);


    utterance.onend = function () {
        // Create a new Blob object with the audio data and type
        var blob = new Blob([new Uint8Array(recordedChunks)], {
            type: 'audio/mp3'
        });

        // Create a new URL object from the Blob object
        var url = URL.createObjectURL(blob);

        // Create a new anchor element to trigger the download
        var anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = 'speech.mp3';
        document.body.appendChild(anchor);
        anchor.click();

        // Clean up the anchor element and URL object
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
    };
}