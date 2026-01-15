from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import SentimentAnalyzer
import pandas as pd
import fitz  # PyMuPDF

# Inisialisasi Aplikasi Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci-rahasia-yang-sangat-aman'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inisialisasi Database dan Login Manager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Model Database untuk User
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Rute untuk Halaman Utama (setelah login)
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# Rute untuk Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login Gagal. Periksa kembali username dan password Anda.', 'danger')
    return render_template('login.html')

# Rute untuk Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Rute untuk memproses analisis

@app.route('/analyze', methods=['POST'])

@login_required

def analyze():

    input_type = request.form.get('input_type')

    source_text = "Tidak diketahui"

    list_to_analyze = []



    try:

        if input_type == 'url':

            source_text = request.form.get('video_url')

            if not source_text:

                flash('Harap masukkan URL YouTube.', 'warning')

                return redirect(url_for('index'))

            

            # Di aplikasi nyata, Anda akan menggunakan API YouTube. Di sini, kita gunakan data dummy.

            list_to_analyze = [

                "Video ini sangat menginspirasi, terima kasih!", "Penjelasannya mudah dipahami, keren banget.",

                "Saya tidak suka dengan konten seperti ini.", "Biasa saja, tidak ada yang spesial.",

                "Kualitas audionya buruk sekali.", "Luar biasa! Saya belajar banyak hal baru.",

                "Cukup bagus, tapi bisa lebih baik lagi.", "Ini adalah video terbaik yang pernah saya tonton!",

                "Pembicaranya kurang jelas saat menyampaikan materi.", "Sangat membosankan, saya tidak menonton sampai selesai."

            ]

        

        elif input_type == 'text':

            raw_text = request.form.get('raw_text')

            source_text = "Input Teks Pengguna"

            if not raw_text or not raw_text.strip():

                flash('Input teks tidak boleh kosong.', 'warning')

                return redirect(url_for('index'))

            list_to_analyze = [line for line in raw_text.strip().split('\n') if line.strip()]



        elif input_type == 'pdf':

            if 'pdf_file' not in request.files or request.files['pdf_file'].filename == '':

                flash('Tidak ada file PDF yang dipilih.', 'warning')

                return redirect(url_for('index'))

            

            pdf_file = request.files['pdf_file']

            if pdf_file and pdf_file.filename.endswith('.pdf'):

                source_text = f"File PDF: {pdf_file.filename}"

                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

                full_text = "".join(page.get_text() for page in doc)

                doc.close()

                list_to_analyze = [line for line in full_text.strip().split('\n') if line.strip()]

            else:

                flash('File tidak valid. Harap unggah file dengan format .pdf', 'danger')

                return redirect(url_for('index'))

        

        if not list_to_analyze:

            flash('Tidak ada teks yang bisa dianalisis dari sumber yang diberikan.', 'info')

            return redirect(url_for('index'))



        # --- Jalankan semua analisis ---

        analyzer = SentimentAnalyzer()

        algorithms = ['svm', 'naive_bayes', 'lexicon']

        all_results = {}



        for algo in algorithms:

            # Lakukan analisis

            results = analyzer.analyze(list_to_analyze, algo)

            results_df = pd.DataFrame(results)

            

            # Siapkan data untuk grafik

            sentiment_counts = results_df['Sentimen'].value_counts()

            chart_labels = sentiment_counts.index.tolist()

            chart_data = sentiment_counts.values.tolist()

            

            # Simpan semua hasil untuk algoritma ini

            all_results[algo] = {

                'df_html': results_df.to_html(classes='data table table-sm table-striped', index=False, border=0),

                'chart_labels': chart_labels,

                'chart_data': chart_data

            }



        return render_template(

            'results.html', 

            source_text=source_text,

            all_results=all_results

        )



    except Exception as e:

        flash(f"Terjadi error saat memproses permintaan: {e}", "danger")

        return redirect(url_for('index'))


if __name__ == '__main__':
    with app.app_context():
        # Buat semua tabel database jika belum ada
        db.create_all()
        # Buat user default jika belum ada
        if not User.query.filter_by(username='admin').first():
            default_user = User(username='admin')
            default_user.set_password('password')
            db.session.add(default_user)
            db.session.commit()
            print("User 'admin' dengan password 'password' telah dibuat.")
            
    app.run(debug=True)