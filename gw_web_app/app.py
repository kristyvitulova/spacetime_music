# app.py

from flask import Flask, render_template, request, jsonify, url_for
from flask_caching import Cache
from waveform_utils import (
    generate_gw_waveform, save_waveform_to_wav, plot_ng_waveform_with_specgram, normalize_waveform
)
import os
import hashlib

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Ensure output directories exist
OUTPUT_DIR = os.path.join(app.root_path, 'static', 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    mass1 = data.get('mass1', 30.0)
    mass2 = data.get('mass2', 30.0)
    spin1x = data.get('spin1x', 0.0)
    spin1y = data.get('spin1y', 0.0)
    spin1z = data.get('spin1z', 0.0)
    spin2x = data.get('spin2x', 0.0)
    spin2y = data.get('spin2y', 0.0)
    spin2z = data.get('spin2z', 0.0)
    sample_rate = data.get('sample_rate', 44100)

    params = {
        'mass1': mass1,
        'mass2': mass2,
        'spin1x': spin1x,
        'spin1y': spin1y,
        'spin1z': spin1z,
        'spin2x': spin2x,
        'spin2y': spin2y,
        'spin2z': spin2z,
        'sample_rate': sample_rate
    }

    # Generate a unique key based on parameters
    params_str = str(params)
    cache_key = hashlib.md5(params_str.encode('utf-8')).hexdigest()

    # Check if results are cached
    cached_result = cache.get(cache_key)
    if cached_result:
        response = {
            'image_url': url_for('static', filename=f'output/{cached_result["image_filename"]}'),
            'audio_url': url_for('static', filename=f'output/{cached_result["audio_filename"]}')
        }
        return jsonify(response)

    # Generate waveform
    try:
        domain, waveform, original_sample_rate = generate_gw_waveform(
            mass1=mass1,
            mass2=mass2,
            spin1x=spin1x,
            spin1y=spin1y,
            spin1z=spin1z,
            spin2x=spin2x,
            spin2y=spin2y,
            spin2z=spin2z
        )
        waveform = waveform[(4, 4)].real
        
        waveform = normalize_waveform(waveform)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # File names based on cache key
    filename_base = f'waveform_{cache_key}'
    image_filename = filename_base + '.png'
    audio_filename = filename_base + '.wav'
    image_filepath = os.path.join(OUTPUT_DIR, image_filename)
    audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)

    # Save waveform image
    plot_ng_waveform_with_specgram(domain, waveform, original_sample_rate, 'Gravitational Waveform', image_filepath)
    # Save audio file
    save_waveform_to_wav(waveform, original_sample_rate, audio_filepath)

    # Cache the result
    cache.set(cache_key, {
        'image_filename': image_filename,
        'audio_filename': audio_filename
    })

    response = {
        'image_url': url_for('static', filename=f'output/{image_filename}'),
        'audio_url': url_for('static', filename=f'output/{audio_filename}')
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
