// static/js/script.js

document.getElementById('params-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const mass1 = parseFloat(document.getElementById('mass1').value);
    const mass2 = parseFloat(document.getElementById('mass2').value);
    const spin1x = parseFloat(document.getElementById('spin1x').value);
    const spin1y = parseFloat(document.getElementById('spin1y').value);
    const spin1z = parseFloat(document.getElementById('spin1z').value);
    const spin2x = parseFloat(document.getElementById('spin2x').value);
    const spin2y = parseFloat(document.getElementById('spin2y').value);
    const spin2z = parseFloat(document.getElementById('spin2z').value);
    const speed_factor = parseFloat(document.getElementById('speed_factor').value);

    const data = {
        mass1: mass1,
        mass2: mass2,
        spin1x: spin1x,
        spin1y: spin1y,
        spin1z: spin1z,
        spin2x: spin2x,
        spin2y: spin2y,
        spin2z: spin2z,
        speed_factor: speed_factor
    };

    fetch('/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            document.getElementById('waveform-image').src = data.image_url + '?t=' + new Date().getTime();
            document.getElementById('waveform-audio').src = data.audio_url + '?t=' + new Date().getTime();
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
