from flask import Flask, render_template, Response,request,flash,url_for,redirect,session
import cv2
from flask_pymongo import PyMongo
from passlib.hash import pbkdf2_sha256
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import logging
from logging.handlers import RotatingFileHandler
import os

video_feed = None
app = Flask(__name__)

app.config['SECRET_KEY'] = '123456'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/your_database_name'

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)


def gen_frames(name):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(f"{name}_classifier.xml")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id, confidence = recognizer.predict(roi_gray)
            confidence = 100 - int(confidence)

            if confidence > 50:
                text = 'Recognized: ' + name.upper()
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text, (x, y - 4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                text = "Unknown Face"
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frame = cv2.putText(frame, text, (x, y - 4), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


mongo = PyMongo(app)
login_manager = LoginManager(app)
login_manager.login_view = 'signin'

# Define the User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

# Callback to reload the user object
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user and pbkdf2_sha256.verify(request.form['password'], existing_user['password']):
            user_obj = User(existing_user['_id'])
            login_user(user_obj)
            flash('Login successful!', 'success')
            return redirect(url_for('dash'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('signin.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user is None:
            hashed_password = pbkdf2_sha256.hash(request.form['password'])
            user_id = users.insert_one({'username': request.form['username'], 'password': hashed_password,'email':request.form['email']}).inserted_id
            user_obj = User(user_id)
            login_user(user_obj)
            flash('Account created successfully!', 'success')
            return redirect(url_for('signin'))
        else:
            flash('Username already exists. Please choose a different one.', 'danger')

    return render_template("signup.html")


@app.route('/dash')
@login_required
def dash():
    return render_template('dash.html', username=current_user.id)

@app.route('/eman')
def eman():
    return render_template('eman.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/detect')
def detect(name):
    path = "./data/" + name
    num_of_images = 0
    detector = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    try:
        os.makedirs(path)
    except:
        print('Directory Already Created')
    vid = cv2.VideoCapture(0)
    while True:

        ret, img = vid.read()
        new_img = None
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(img, "Face Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
            cv2.putText(img, str(str(num_of_images) + " images captured"), (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255))
            new_img = img[y:y + h, x:x + w]
        cv2.imshow("Face Detection", img)
        key = cv2.waitKey(1) & 0xFF

        try:
            cv2.imwrite(str(path + "/" + str(num_of_images) + name + ".jpg"), new_img)
            num_of_images += 1
        except:

            pass
        if key == ord("q") or key == 27 or num_of_images > 300:  # take 300 frames
            break
    cv2.destroyAllWindows()
    return num_of_images


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        address = request.form.get('subject')
        phn = request.form.get('message')
        u_id = request.form.get('u_id')

        data = {
            'name': name,
            'email': email,
            'address': address,
            'phone no': phn,
            'ID': u_id
        }
        try:
            mongo.db.feedback_report.insert_one(data)
            flash("Feedback Submitted Successfully", 'success')
        except Exception as e:
            flash("Error submitting form", 'danger')

        return redirect(url_for('dash'))

    # If it's a GET request, you might want to render a template or return a response.
    return render_template('feedback.html')  

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/start_cam', methods=['POST'])
def start_cam():
    name = request.form.get('name')
    global video_feed
    video_feed = f"/video_feed/{name}"
    return render_template('eman.html', video_src=video_feed)

@app.route('/stop_cam', methods=['POST'])
def stop_cam():
    global video_feed
    video_feed = None
    return render_template('eman.html', video_src=video_feed)

@app.route('/video_feed/<name>')
def video_feed(name):
    return Response(gen_frames(name), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
