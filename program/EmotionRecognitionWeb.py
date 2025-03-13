import os
import uuid
import re
import numpy as np
import cv2
import io
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from deepface import DeepFace
from PIL import Image
import json


app = Flask(__name__, template_folder="templates")
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')  # Используем переменные окружения
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)





@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    image_path = db.Column(db.String(255), nullable=False)
    dominant_emotion = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    emotions = db.Column(db.Text, nullable=False)  # Хранение эмоций в JSON-формате

    user = db.relationship('User', backref=db.backref('results', lazy=True))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Проверяем, все ли данные введены
        if not username or not password or not confirm_password:
            flash('Все поля обязательны для заполнения.', 'danger')
            return redirect(url_for('register'))

        # Проверка сложности пароля
        if len(password) < 6 or not any(char.isdigit() for char in password) or not any(
                char.isalpha() for char in password):
            flash('Пароль должен быть длиной минимум 6 символов, содержать буквы и цифры.', 'danger')
            return redirect(url_for('register'))

        # Проверяем совпадение паролей
        if password != confirm_password:
            flash('Пароли не совпадают. Попробуйте снова.', 'danger')
            return redirect(url_for('register'))

        # Проверяем существование имени пользователя
        if User.query.filter_by(username=username).first():
            flash('Пользователь с таким именем уже существует.', 'danger')
            return redirect(url_for('register'))

        # Создаём нового пользователя
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Вы успешно зарегистрировались! Теперь войдите.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')  # Замените email_or_username на username
        password = request.form.get('password')

        # Находим пользователя только по username
        user = User.query.filter_by(username=username).first()

        # Проверяем, существует ли пользователь и совпадает ли пароль
        if user and user.check_password(password):
            login_user(user)
            flash('Вы успешно вошли!', 'success')
            return redirect(url_for('home'))

        # Неверные данные
        flash('Неверные данные для входа. Проверьте имя пользователя или пароль.', 'danger')

    return render_template('login.html')




@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Вы вышли из системы.', 'success')
    return redirect(url_for('login'))


def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)


def convert_numpy(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


def normalize_emotions(emotion_dict):
    total = sum(emotion_dict.values())
    if total > 0:
        return {emotion: (value / total) * 100 for emotion, value in emotion_dict.items()}
    return emotion_dict


@app.route('/')
@login_required
def home():
    return render_template('base.html', title="Welcome")


@app.route('/image_analysis')
@login_required
def image_analysis():
    return render_template('image_analysis.html', title="Image Analysis")


@app.route('/real_time_analysis')
@login_required
def real_time_analysis():
    return render_template('real_time_analysis.html', title="Real-time Analysis")


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    file_name = sanitize_filename(str(uuid.uuid4()) + "_" + file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    file.save(file_path)

    try:
        # Анализ изображения на наличие эмоций для всех лиц
        results = DeepFace.analyze(file_path, actions=['emotion'], enforce_detection=False)

        if not isinstance(results, list):
            results = [results]

        img = cv2.imread(file_path)
        all_emotions = []

        for result in results:
            region = result.get("region")
            if region:
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                emotion_dict = result["emotion"]

                # Нормализуем эмоции
                normalized_emotions = normalize_emotions(emotion_dict)
                emotion = max(normalized_emotions, key=normalized_emotions.get)
                confidence = normalized_emotions[emotion]

                all_emotions.append({
                    "region": region,
                    "dominant_emotion": emotion,
                    "confidence": confidence,
                    "emotions": normalized_emotions
                })

                # Сохраняем результат в базу данных
                result_record = Result(
                    user_id=current_user.id,
                    image_path=file_path,  # Сохраняем путь к изображению
                    dominant_emotion=emotion,
                    confidence=confidence,
                    emotions=json.dumps(convert_numpy(normalized_emotions))  # Преобразуем эмоции в строку JSON
                )
                db.session.add(result_record)

                # Рисуем прямоугольник вокруг лица
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{emotion} ({confidence:.1f}%)"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        db.session.commit()  # Сохраняем изменения в базу данных

        # Сохраняем обработанное изображение
        result_image_path = os.path.join(RESULTS_FOLDER, file_name)
        cv2.imwrite(result_image_path, img)

        results_serializable = convert_numpy(all_emotions)

        return jsonify({
            "results": results_serializable,
            "result_image_url": f"/results/{file_name}"
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            pass




@app.route('/analyze_realtime', methods=['POST'])
@login_required
def analyze_realtime():
    """Анализ эмоций нескольких людей в реальном времени"""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    try:
        # Загружаем изображение и преобразуем его для обработки с помощью OpenCV
        image = Image.open(io.BytesIO(file.read()))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Анализируем изображение с использованием DeepFace
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Убедимся, что результат всегда список (даже если одна эмоция)
        if not isinstance(results, list):
            results = [results]

        if not results:
            return jsonify({"error": "No result from DeepFace"}), 400

        all_emotions = []
        for result in results:
            region = result.get("region")
            if region:
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                emotion_dict = result["emotion"]

                # Нормализуем эмоции
                normalized_emotions = normalize_emotions(emotion_dict)

                dominant_emotion = max(normalized_emotions, key=normalized_emotions.get)
                confidence = round(normalized_emotions[dominant_emotion], 2)

                all_emotions.append({
                    "region": region,
                    "dominant_emotion": dominant_emotion,
                    "confidence": confidence,
                    "emotions": normalized_emotions
                })

        # Преобразуем результат в сериализуемый формат
        results_serializable = convert_numpy(all_emotions)

        return jsonify({
            "results": results_serializable
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route('/history')
@login_required
def history():
    user_id = current_user.id
    results = Result.query.filter_by(user_id=user_id).all()

    # Преобразование JSON-строки в словарь
    for result in results:
        if isinstance(result.emotions, str):
            result.emotions = json.loads(result.emotions)

    return render_template('history.html', results=results)


@app.route('/results/<path:filename>')
def serve_result_image(filename):
    """Отдает обработанные изображения"""
    return send_file(os.path.join(RESULTS_FOLDER, filename))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)


















