from flask import Flask, render_template, redirect, request, url_for, flash, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
import pandas as pd
import os
import re
import string
import pickle
import math
from functools import wraps
from werkzeug.utils import secure_filename
from urllib.parse import unquote
import io


# NLP Libraries
try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

try:
    from nlp_id import Lemmatizer, StopWord
    NLP_ID_AVAILABLE = True
except Exception:
    try:
        from nlp_id.lemmatizer import Lemmatizer
        from nlp_id.stopword import StopWord
        NLP_ID_AVAILABLE = True
    except Exception:
        NLP_ID_AVAILABLE = False

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# =====================
# APP INIT
# =====================
app = Flask(__name__)
app.secret_key = "skripsi_emas"

# CORS — agar bisa diakses dari tunnel / domain berbeda
if CORS_AVAILABLE:
    CORS(app, resources={r"/api/*": {"origins": "*"}})

# =====================
# DATABASE
# =====================
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/skripsi_emas'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# =====================
# FOLDER SETUP
# =====================
UPLOAD_FOLDER = 'dataset'
MODEL_FOLDER  = 'model'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH   = os.path.join(MODEL_FOLDER, "model_pipeline.pkl")

def get_dataset_file():
    """
    Kembalikan path file dataset yang aktif, urutan prioritas:
    1. File yang dipilih user via 'Pilih & Proses' (session selected_file)
    2. Hasil preprocessing (hasil_preprocessing.csv)
    3. File CSV manapun yang ada di folder dataset (ambil yang pertama)
    Kembalikan None jika tidak ada sama sekali.
    """
    from flask import session as _session

    # 1. File yang dipilih user
    fname = _session.get('selected_file')
    if fname:
        path = os.path.join(UPLOAD_FOLDER, fname)
        if os.path.exists(path):
            return path

    # 2. Hasil preprocessing
    preproc = os.path.join(UPLOAD_FOLDER, 'hasil_preprocessing.csv')
    if os.path.exists(preproc):
        return preproc

    # 3. File CSV pertama yang ada di folder
    try:
        files = sorted([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')])
        if files:
            return os.path.join(UPLOAD_FOLDER, files[0])
    except Exception:
        pass

    return None

# =====================
# GLOBAL STATE
# =====================
temp_df          = None
current_file     = None
_pipeline_cache  = None  # Cache model agar tidak load dari disk setiap request

def _get_pipeline():
    """Load model sekali lalu cache di memori; tidak baca disk lagi sampai model baru ditraining."""
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        _pipeline_cache = pickle.load(f)
    return _pipeline_cache

def _get_temp_df():
    """Ambil temp_df dari memori; kalau kosong, reload dari file yang tercatat di session."""
    global temp_df, current_file
    if temp_df is not None:
        return temp_df
    # Coba reload dari session
    from flask import session as _session
    fname = _session.get('selected_file')
    if fname:
        path = os.path.join(UPLOAD_FOLDER, fname)
        if os.path.exists(path):
            temp_df      = read_csv_smart(path)
            current_file = fname
            return temp_df
    return None

# =====================
# LOGIN REQUIRED DECORATOR
# =====================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("Silakan login terlebih dahulu!")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# =====================
# KAMUS EMOJI MANUAL
# =====================
KAMUS_EMOJI_MANUAL = {
    # Positif
    "😍": "sangat suka luar biasa", "🥰": "sangat senang suka",
    "😊": "senang bahagia", "😁": "senang gembira", "😄": "bahagia gembira",
    "🤩": "sangat kagum luar biasa", "👍": "bagus setuju oke",
    "👏": "bagus hebat apresiasi", "🎉": "selamat merayakan sukses",
    "✅": "oke benar sesuai", "💯": "sempurna seratus persen",
    "🔥": "luar biasa keren", "💪": "kuat semangat", "🏆": "juara menang sukses",
    "🥇": "juara emas terbaik", "⭐": "bintang bagus", "🌟": "bintang bagus istimewa",
    "💎": "berharga mewah", "💰": "untung profit", "📈": "naik untung profit",
    "🚀": "naik cepat melesat", "😎": "keren percaya diri", "😉": "senang bercanda",
    # Negatif
    "😡": "marah sangat kesal", "🤬": "sangat marah frustrasi",
    "😤": "kesal frustrasi dongkol", "😠": "marah kesal",
    "😢": "sedih kecewa", "😭": "sangat sedih menangis kecewa",
    "😱": "kaget panik takut", "😰": "khawatir panik", "😨": "takut cemas",
    "😩": "frustrasi menyerah", "😫": "lelah frustrasi", "🤦": "malu kecewa",
    "👎": "jelek tidak setuju buruk", "❌": "salah tidak gagal",
    "⛔": "dilarang gagal", "🚫": "tidak boleh gagal",
    "💸": "rugi kehilangan uang boros", "📉": "turun rugi merosot",
    "😞": "kecewa sedih", "😔": "sedih kecewa menyesal", "😒": "kecewa tidak puas",
    "☹️": "sedih kecewa",
    # Netral
    "🤔": "berpikir mempertimbangkan", "🧐": "menganalisis mempertimbangkan",
    "💡": "ide gagasan informasi", "📊": "data statistik analisis",
    "📋": "catatan informasi", "ℹ️": "informasi keterangan",
    "📢": "pengumuman informasi", "🔔": "notifikasi pengingat",
    "💬": "komentar diskusi", "🙏": "mohon terima kasih",
    "👀": "melihat memperhatikan", "🤷": "tidak tahu bingung",
    "⚠️": "peringatan hati-hati waspada", "❓": "bertanya tanya",
    "❗": "penting perhatian",
}

# =====================
# PREPROCESSOR CLASS
# =====================
class PreprocessorTeks:
    def __init__(self):
        if NLP_ID_AVAILABLE:
            self.lemmatizer = Lemmatizer()
            self.stopword_remover = StopWord()
        else:
            self.lemmatizer = None
            self.stopword_remover = None

        if SASTRAWI_AVAILABLE:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        else:
            self.stemmer = None

        # ── OPTIMASI: cache stem — kata sama tidak di-stem ulang ──────────────
        self._stem_cache: dict = {}

        # ── OPTIMASI: compile semua regex saat startup, bukan per kalimat ─────
        self._re_nonlatin   = re.compile(r'[^\x00-\x7F\u00C0-\u024F\s]')
        self._re_ws         = re.compile(r'\s+')
        self._re_br         = re.compile(r'<br\s*/?>', re.IGNORECASE)
        self._re_tag        = re.compile(r'<[^>]+>')
        self._re_entity     = re.compile(r'&#\d+;')
        self._re_digit      = re.compile(r'\d+')
        self._re_repeat     = re.compile(r'(.)\1{2,}')
        self._re_exclaim    = re.compile(r'[!]{2,}')
        self._re_question   = re.compile(r'[?]{2,}')
        self._re_angka      = re.compile(
            r'\b\d+\s*(?:gram|gr|karat|tahun|bulan|hari|'
            r'rupiah|rb|ribu|jt|juta|persen|kg|mg)(?!\w)'
            r'|\b\d+\s*%'
            r'|\b(?:19|20)\d{2}\b'
            r'|\b(?:emas|gold)\s+\d{1,2}\b'
            r'|\b\d{1,2}\s*(?:karat|k)\b'
            r'|\b(?:galeri|toko|butik|outlet|pusat|gerai|kios|studio|antam|pegadaian)\s+\d+\b'
            r'|\b\d+\s*(?:galeri|toko)\b'
            r'|\bRp\.?\s*\d[\d.,]*'
            r'|\b\d[\d.,]*\s*(?:per\s*gram|/gram|per\s*gr|/gr)\b',
            re.IGNORECASE
        )
        self._tbl_punct = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

        # compile regex emoji ascii
        self._re_emoji_ascii = [
            (re.compile(r'[:;=8][\-o\*\']?[\)\]\}]'), ' senang '),
            (re.compile(r'[:;=8][\-o\*\']?[\(\[\{]'), ' sedih '),
            (re.compile(r'[:;=8][\-o\*\']?[Dd]'),     ' senyum '),
            (re.compile(r'[xX][\-]?[Dd]'),             ' tertawa '),
            (re.compile(r'<3'),                         ' cinta '),
            (re.compile(r'T[\._]T'),                    ' menangis '),
        ]

        # compile 40+ regex sarkasme — early-exit saat pertama cocok
        self._re_sarkasme = [
            (re.compile(pola, re.IGNORECASE), ganti)
            for pola, ganti in self.KAMUS_SARKASME.items()
        ]

        # compile regex normalisasi data
        self._re_normdata = [
            (re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE), v)
            for k, v in self.NORMALISASI_DATA.items()
        ]

    def konversi_emoji(self, teks):
        if pd.isna(teks):
            return ""
        teks_out = str(teks)
        for em, deskripsi in KAMUS_EMOJI_MANUAL.items():
            teks_out = teks_out.replace(em, f' {deskripsi} ')
        if EMOJI_AVAILABLE:
            try:
                teks_out = emoji.replace_emoji(teks_out, replace='')
            except Exception:
                pass
        teks_out = self._re_nonlatin.sub(' ', teks_out)
        return self._re_ws.sub(' ', teks_out).strip()

    def bersihkan_teks(self, teks):
        teks = self._re_br.sub(' ', teks)
        teks = self._re_tag.sub(' ', teks)
        teks = teks.replace('&amp;', '&').replace('&lt;', '<') \
                   .replace('&gt;', '>').replace('&nbsp;', ' ')
        teks = self._re_entity.sub('', teks)

        # ── OPTIMASI: re.sub callback — satu pass, tanpa loop str.replace ─────
        KODE_HURUF = [
            'ALFA','BETA','GAMA','DELTA','EPSILON','ZETA','ETA','THETA',
            'IOTA','KAPPA','LAMBDA','MU','NU','XI','OMIKRON','PI',
            'RHO','SIGMA','TAU','UPSILON','PHI','KHI','PSI','OMEGA',
            'ALEF','BET','GIMEL','DALET','HE','VAV','ZAYIN','KHET',
        ]
        counter = [0]
        saved   = {}

        def _simpan(m):
            key = f'__{KODE_HURUF[counter[0] % len(KODE_HURUF)]}__'
            while key in saved:
                counter[0] += 1
                key = f'__{KODE_HURUF[counter[0] % len(KODE_HURUF)]}X__'
            saved[key] = m.group()
            counter[0] += 1
            return key

        teks = self._re_angka.sub(_simpan, teks)
        teks = self._re_digit.sub('', teks)
        for key, value in saved.items():
            teks = teks.replace(key, value)

        teks = teks.translate(self._tbl_punct)
        return self._re_ws.sub(' ', teks).strip()

    def case_folding(self, teks):
        return str(teks).lower()

    KAMUS_SLANG = {
    # === KATA GANTI ORANG (original) ===
    "sya": "saya", "sy": "saya", "aq": "aku", "gw": "saya",
    "gue": "saya", "lo": "kamu", "lu": "kamu", "lw": "kamu",
    "mrk": "mereka", "kt": "kita", "km": "kamu",
    "doi": "dia", "dy": "dia", "dia": "dia", "dya": "dia",
    "org": "orang", "orng": "orang", "orang2": "orang-orang",
    "tmn": "teman", "tmen": "teman", "sobat": "teman", "gaes": "teman-teman",
    "guys": "teman-teman", "bro": "saudara", "sis": "saudari",
    "kak": "kakak", "kk": "kakak", "ko": "kakak", "ka": "kakak",
    "bang": "abang", "bg": "abang", "mas": "mas", "mba": "mbak",

    # === NEGASI (original + tambahan) ===
    "gk": "tidak", "gak": "tidak", "nggak": "tidak", "ga": "tidak",
    "ngga": "tidak", "tdk": "tidak", "enggak": "tidak", "ndak": "tidak",
    "tak": "tidak", "g": "tidak", "kagak": "tidak", "kaga": "tidak",
    "nope": "tidak", "nop": "tidak", "gakda": "tidak ada",
    "gaada": "tidak ada", "ga ada": "tidak ada", "gada": "tidak ada",
    "blum": "belum", "belom": "belum",

    # === INTENSIFIER (original + tambahan) ===
    "bgt": "sangat", "bgtt": "sangat", "bnget": "sangat",
    "banget": "sangat", "amat": "sangat", "pol": "sangat",
    "abis": "sangat", "habis": "sangat", "parah": "sangat",
    "amit": "sangat", "bet": "sangat", "bener": "benar",
    "bnr": "benar", "emg": "memang", "emang": "memang",
    "memang": "memang", "sbnrnya": "sebenarnya", "sebenernya": "sebenarnya",
    "aslinya": "sebenarnya", "jujur": "sebenarnya",

    # === KATA HUBUNG & PARTIKEL (original + tambahan) ===
    "yg": "yang", "yng": "yang", "dgn": "dengan", "dg": "dengan",
    "bbrp": "beberapa", "tp": "tapi", "tpi": "tapi",
    "adlh": "adalah", "scr": "secara", "sbg": "sebagai",
    "dr": "dari", "jg": "juga", "jga": "juga",
    "klo": "kalau", "klu": "kalau", "kl": "kalau", "kalo": "kalau",
    "klau": "kalau", "bila": "jika", "kpn": "kapan", "kapn": "kapan",
    "gmn": "bagaimana", "gmana": "bagaimana", "gimana": "bagaimana",
    "knp": "mengapa", "kenapa": "mengapa", "ngapa": "mengapa",
    "pdhl": "padahal", "phdl": "padahal", "wlpn": "walaupun",
    "mskpn": "meskipun", "walopun": "walaupun", "wlopun": "walaupun",
    "krn": "karena", "krna": "karena", "karna": "karena", "soalnya": "karena",
    "soal": "karena", "makanya": "oleh karena itu", "jdnya": "jadinya",
    "jdinya": "jadinya", "akhirnya": "akhirnya", "trnyata": "ternyata",
    "nyatanya": "ternyata", "pdahal": "padahal",

    # === KATA KERJA UMUM (original + tambahan) ===
    "udah": "sudah", "udh": "sudah", "sdh": "sudah", "dah": "sudah",
    "blm": "belum", "blom": "belum", "msh": "masih", "masi": "masih",
    "hrs": "harus", "utk": "untuk", "tuk": "untuk",
    "lgsg": "langsung", "lngsng": "langsung", "langsung": "langsung",
    "jd": "jadi", "jdi": "jadi", "bs": "bisa", "bsa": "bisa",
    "mo": "mau", "mao": "mau", "pgn": "ingin", "pengen": "ingin",
    "pingin": "ingin", "inginn": "ingin", "kepengen": "ingin",
    "coba": "mencoba", "nyoba": "mencoba", "cobain": "mencoba",
    "pake": "menggunakan", "make": "menggunakan", "makai": "menggunakan",
    "pakein": "menggunakan", "dipake": "digunakan", "dipakai": "digunakan",
    "beli": "membeli", "beli2": "membeli-beli", "nyari": "mencari",
    "cari": "mencari", "dpt": "dapat", "dapet": "mendapatkan",
    "dpet": "mendapatkan", "dapetin": "mendapatkan", "dapatin": "mendapatkan",
    "jual": "menjual", "jualin": "menjual", "dijual": "dijual",
    "simpen": "menyimpan", "nyimpen": "menyimpan", "taruh": "menyimpan",
    "taro": "menaruh", "naro": "menaruh", "kasih": "memberi",
    "ngasih": "memberi", "dikasih": "diberikan", "nanya": "bertanya",
    "tanya": "bertanya", "nanyain": "menanyakan", "jawab": "menjawab",
    "ngomong": "berbicara", "bilang": "mengatakan", "blng": "mengatakan",
    "ngitung": "menghitung", "itung": "menghitung", "hitung": "menghitung",
    "ngerti": "mengerti", "ngerti": "mengerti", "paham": "memahami",
    "tau": "tahu", "tw": "tahu", "taw": "tahu", "gatau": "tidak tahu",
    "gatw": "tidak tahu", "entah": "tidak tahu",
    "liat": "melihat", "liet": "melihat", "lihatin": "melihat",
    "ngeliat": "melihat", "tengok": "melihat",
    "denger": "mendengar", "dengerin": "mendengarkan",
    "ngedenger": "mendengar", "dngr": "mendengar",
    "nunggu": "menunggu", "tunggu": "menunggu", "nungguin": "menunggu",
    "cair": "mencairkan", "cairkan": "mencairkan", "dicairkan": "dicairkan",
    "tarik": "menarik", "narik": "menarik", "tarik tunai": "penarikan tunai",

    # === WAKTU (original + tambahan) ===
    "jt": "juta", "rb": "ribu", "bln": "bulan", "thn": "tahun",
    "gr": "gram", "kg": "kilogram",
    "skrng": "sekarang", "skrg": "sekarang", "skr": "sekarang",
    "skarang": "sekarang", "stlh": "setelah", "stelah": "setelah",
    "sblm": "sebelum", "sebelom": "sebelum", "abis ini": "setelah ini",
    "abisnya": "setelahnya", "dulu": "dahulu", "dl": "dahulu",
    "duluan": "lebih dahulu", "bentar": "sebentar", "btar": "sebentar",
    "sebentar": "sebentar", "lama": "lama", "lmyn": "lumayan",
    "lumayan": "lumayan", "lamaaa": "sangat lama",

    # === EMOSI POSITIF (original + tambahan) ===
    "mantap": "bagus", "mantul": "bagus", "mantabs": "bagus", "mantab": "bagus",
    "keren": "bagus", "kereen": "bagus", "kece": "bagus", "kece badai": "sangat bagus",
    "gokil": "luar biasa", "gilak": "luar biasa", "gila": "luar biasa",
    "gilaa": "luar biasa", "asik": "menyenangkan", "asyik": "menyenangkan",
    "seru": "menyenangkan", "wah": "kagum", "wow": "kagum", "woah": "kagum",
    "kuy": "ayo", "gaskeun": "ayo lakukan", "gaspol": "ayo semangat",
    "semangat": "bersemangat", "smangat": "bersemangat",
    "alhamdulillah": "syukurlah", "syukur": "syukurlah",
    "seneng": "senang", "senengg": "senang", "happy": "bahagia",
    "bahagia": "bahagia", "puas": "puas", "puass": "sangat puas",
    "recommended": "direkomendasikan", "rekomen": "direkomendasikan",
    "worth": "sepadan", "worthit": "sepadan", "worth it": "sepadan",

    # === EMOSI NEGATIF (original + tambahan) ===
    "nyesel": "menyesal", "nyesal": "menyesal", "menyesal": "menyesal",
    "sebel": "kesal", "sebal": "kesal", "bete": "kesal", "bt": "kesal",
    "kesel": "kesal", "keseeel": "sangat kesal", "jengkel": "kesal",
    "gondok": "kesal", "dongkol": "kesal", "sewot": "marah",
    "marah": "marah", "emosi": "marah", "esmosi": "marah",
    "benci": "membenci", "bete": "tidak suka", "males": "malas",
    "mls": "malas", "mager": "malas gerak", "males bgt": "sangat malas",
    "cape": "lelah", "capek": "lelah", "capee": "sangat lelah",
    "kecewa": "kecewa", "kcwa": "kecewa", "kecewaa": "sangat kecewa",
    "ketipu": "tertipu", "ktipu": "tertipu", "ditipu": "tertipu",
    "rugi": "rugi", "rugiii": "sangat rugi", "boncos": "rugi besar",
    "jebol": "gagal", "bangkrut": "bangkrut", "kapok": "jera",
    "kapook": "sangat jera", "tobat": "jera",
    "sedih": "sedih", "sedihh": "sangat sedih", "galau": "bingung sedih",
    "pusing": "bingung", "bingung": "bingung", "ragu": "ragu-ragu",
    "curiga": "curiga", "waspada": "waspada", "takut": "takut",
    "parno": "paranoid", "khawatir": "khawatir",

    # === KATA SIFAT UMUM (original + tambahan) ===
    "mending": "lebih baik", "mendingan": "lebih baik", "断然": "lebih baik",
    "ribet": "rumit", "susah": "sulit", "susyah": "sulit",
    "gampang": "mudah", "gmpng": "mudah", "simpel": "sederhana",
    "cepet": "cepat", "cpet": "cepat", "lelet": "lambat",
    "lemot": "lambat", "lambat": "lambat", "telat": "terlambat",
    "mahal": "mahal", "mihil": "mahal", "murah": "murah",
    "murce": "murah", "hemat": "hemat", "irit": "hemat",
    "untung": "menguntungkan", "cuan": "menguntungkan", "profit": "keuntungan",
    "rugi": "merugikan", "loss": "rugi", "minus": "rugi",
    "aman": "aman", "save": "aman", "safe": "aman",
    "bahaya": "berbahaya", "berbahaya": "berbahaya", "riskan": "berisiko",
    "berisiko": "berisiko", "riskee": "berisiko",
    "legal": "resmi", "ilegal": "tidak resmi", "resmi": "resmi",
    "abal": "palsu", "abal2": "palsu", "palsu": "palsu",
    "ori": "asli", "original": "asli", "kw": "palsu tiruan",
    "zonk": "gagal tidak sesuai", "scam": "penipuan", "fraud": "penipuan",
    "hoax": "informasi palsu", "bohong": "berbohong", "boong": "berbohong",
    "tipu": "menipu", "nipu": "menipu", "penipu": "penipu",

    # === SAPAAN & PENUTUP KOMENTAR ===
    "makasih": "terima kasih", "mksh": "terima kasih", "tks": "terima kasih",
    "thx": "terima kasih", "thanks": "terima kasih", "ty": "terima kasih",
    "maturnuwun": "terima kasih", "hatur nuhun": "terima kasih",
    "ok": "oke", "oke": "oke", "sip": "oke", "siip": "oke",
    "siap": "oke", "setuju": "setuju", "sepakat": "setuju",
    "fix": "pasti", "deal": "setuju", "gas": "ayo",

    # === TAWA & EKSPRESI ===
    "wkwk": "tertawa", "wkwkwk": "tertawa", "wkwkwkwk": "tertawa",
    "haha": "tertawa", "hahaha": "tertawa", "hahahaha": "tertawa",
    "hihi": "tertawa", "hehe": "tertawa", "huhu": "menangis",
    "hiks": "menangis", "hikss": "menangis", ":(": "sedih",
    ":)": "senang", "xD": "tertawa", "lol": "tertawa",
    "lmao": "tertawa keras", "rofl": "tertawa keras",

    # === KEUANGAN & INVESTASI UMUM (original + tambahan) ===
    "jt": "juta", "rb": "ribu", "bln": "bulan", "thn": "tahun",
    "nabung": "menabung", "invst": "investasi", "invest": "investasi",
    "investasi": "investasi", "inves": "investasi",
    "dca": "beli bertahap", "dollar cost averaging": "beli bertahap",
    "buyback": "beli kembali", "buy back": "beli kembali",
    "spread": "selisih harga", "fee": "biaya", "admin": "biaya administrasi",
    "bunga": "bunga", "imbal hasil": "keuntungan", "return": "keuntungan",
    "portofolio": "portofolio", "porto": "portofolio",
    "aset": "aset", "modal": "modal", "dana": "dana",
    "likuid": "mudah dicairkan", "likuiditas": "kemudahan pencairan",
    "cicil": "mencicil", "nyicil": "mencicil", "kredit": "kredit",
    "gadai": "menggadaikan", "gadein": "menggadaikan",
    "lelang": "dilelang", "dilelang": "dilelang",
    "laba": "keuntungan", "dividen": "dividen", "yield": "imbal hasil",
    "inflasi": "inflasi", "deflasi": "deflasi", "resesi": "resesi",

    # === EMAS DIGITAL SPESIFIK (original + tambahan) ===
    "bullion": "emas batangan", "antam": "emas antam",
    "lm": "logam mulia", "logam mulia": "logam mulia",
    "emas digital": "emas digital", "emas fisik": "emas fisik",
    "emas batangan": "emas batangan", "emas perhiasan": "emas perhiasan",
    "cetak": "mencetak emas", "cetak emas": "mencetak emas fisik",
    "kadar": "kadar kemurnian", "karat": "karat", "24k": "24 karat murni",
    "22k": "22 karat", "18k": "18 karat",
    "pegadaian": "pegadaian", "antam": "antam", "pluang": "pluang",
    "indogold": "indogold", "lakuemas": "lakuemas", "bukaemas": "bukaemas",
    "tokemas": "tokopedia emas", "shopee emas": "shopee emas",
    "tabungan emas": "tabungan emas", "rekening emas": "rekening emas",
    "sdb": "safe deposit box", "brankas": "brankas",
    "sertifikat": "sertifikat emas", "sertif": "sertifikat",
    "ojk": "otoritas jasa keuangan", "lps": "lembaga penjamin simpanan",
    "berlisensi": "berlisensi resmi", "terdaftar": "terdaftar resmi",
    "harga beli": "harga beli", "harga jual": "harga jual",
    "harga spot": "harga pasar emas", "spot price": "harga pasar emas",

    # === TEKNOLOGI & APLIKASI ===
    "app": "aplikasi", "apk": "aplikasi", "aplk": "aplikasi",
    "download": "unduh", "update": "perbarui", "upgrade": "tingkatkan",
    "login": "masuk", "logout": "keluar", "daftar": "mendaftar",
    "regis": "mendaftar", "verif": "verifikasi", "kyc": "verifikasi identitas",
    "otp": "kode verifikasi", "pin": "nomor identifikasi pribadi",
    "transfer": "transfer", "tf": "transfer", "trf": "transfer",
    "rek": "rekening", "rekening": "rekening", "no rek": "nomor rekening",
    "va": "virtual account", "ewallet": "dompet digital",
    "gopay": "gopay", "ovo": "ovo", "dana": "dana", "shopeepay": "shopeepay",
    "error": "error", "eror": "error", "bug": "kesalahan sistem",
    "gangguan": "gangguan sistem", "maintenance": "pemeliharaan sistem",
    "server down": "server tidak aktif", "down": "tidak aktif",

    # === SINGKATAN MEDIA SOSIAL ===
    "yutub": "youtube", "yt": "youtube", "ig": "instagram",
    "fb": "facebook", "tt": "tiktok", "wa": "whatsapp",
    "dm": "pesan langsung", "pm": "pesan pribadi",
    "komen": "komentar", "koment": "komentar", "reply": "balas",
    "share": "bagikan", "like": "suka", "subscribe": "berlangganan",
    "sub": "berlangganan", "follow": "mengikuti", "unfollow": "berhenti mengikuti",
    "viral": "viral", "trending": "sedang tren",
    }

    KAMUS_SARKASME = {
    # === POLA AWAL (original) ===
        r"bagus\s*(?:sekali|banget|amat|bgt)?\s*(?:ya|sih|nih)?\s*(?:sampe|sampai)\s*(.+)": "sangat buruk",
        r"mantap\s*(?:sekali|banget)?\s*(?:ya|sih)?\s*(?:sampe|sampai)\s*(.+)": "sangat buruk",
        r"bagus\s*(?:banget)?\s*(?:ya|nih|sih)\s*(?:,|\.|\!)?(?:\s*padahal|\s*tapi|\s*tp)": "tidak bagus",
        r"keren\s*(?:banget)?\s*(?:ya|nih|sih)\s*(?:,|\.|\!)?\s*(?:padahal|tapi|tp)": "tidak bagus",
        r"emang\s+(?:terbaik|terpercaya|bagus)\s+(?:ya|nih|sih|dong)\s*(?:,|!|\.)\s*(?:tipu|bohong|rugi|zonk)": "penipuan mengecewakan",
        r"(?:terima kasih|makasih|thx)\s+(?:ya|nih|sih)?\s*(?:udah|sudah)\s+(?:tipu|bohong|rugi|kecewakan)": "sangat mengecewakan",
        r"luar\s+biasa\s+(?:banget\s+)?(?:ya|nih|sih)?\s*(?:sampe|sampai)\s+(?:rugi|kecewa|tipu)": "sangat buruk",

        # === POLA PUJIAN + KONTRADIKSI ===
        r"(?:bagus|keren|mantap|hebat|canggih)\s*(?:banget|bgt|sekali|amat)?\s*(?:ya|sih|nih|dong)?\s*(?:,|\.|\!)?\s*(?:padahal|tapi|tp|tetapi|namun)\s+(?:bohong|tipu|rugi|zonk|scam|penipuan)": "tidak bagus",
        r"(?:luar biasa|amazing|wow|wah)\s*(?:banget)?\s*(?:ya|sih|nih)?\s*(?:padahal|tapi|tp)\s+(.+)": "tidak bagus",
        r"(?:keren|bagus|mantap|hebat)\s*(?:banget|bgt)?\s*(?:ya|sih|nih)\s*(?:,|!)?\s*(?:uang|duit|modal)\s+(?:hilang|lenyap|raib|amblas)": "sangat merugikan",
        r"(?:recommended|rekomen|rekomendasi)\s*(?:banget|bgt)?\s*(?:ya|nih|sih)?\s*(?:buat|untuk)\s+(?:yang mau rugi|yang mau kecewa|yang mau ditipu)": "tidak direkomendasikan",
        r"profesional\s*(?:banget|bgt|sekali)?\s*(?:ya|sih|nih)?\s*(?:sampe|sampai|padahal|tapi)\s+(?:kabur|lari|hilang|tipu|bohong)": "tidak profesional",

        # === POLA TERIMA KASIH SARKASTIK ===
        r"(?:terima kasih|makasih|thx|thanks)\s*(?:ya|nih|sih|banget|bgt)?\s*(?:udah|sudah|telah)?\s*(?:bikin|buat|jadiin|jadikan)\s+(?:rugi|kecewa|bangkrut|susah|sengsara)": "sangat mengecewakan",
        r"(?:terima kasih|makasih|thx)\s*(?:ya|nih|sih)?\s*(?:investasinya|platformnya|aplikasinya)\s*(?:udah|sudah)?\s*(?:bikin|buat)\s+(?:rugi|kecewa|miskin|bangkrut)": "sangat mengecewakan",
        r"(?:makasih|terima kasih|thx)\s*(?:loh|lho|ya)\s*(?:udah|sudah)\s*(?:nipu|bohongin|kecohin|rugiin)": "sangat mengecewakan",

        # === POLA IRONI WAKTU / PROSES ===
        r"(?:cepet|cepat|kilat)\s*(?:banget|bgt|amat|sekali)?\s*(?:ya|sih|nih)?\s*(?:prosesnya|cairnya|responnya)?\s*(?:sampe|sampai)\s*(?:\d+\s*(?:hari|minggu|bulan)|lama|berbulan|bertahun)": "sangat lambat",
        r"(?:gampang|mudah|simpel)\s*(?:banget|bgt|sekali)?\s*(?:ya|sih|nih)?\s*(?:padahal|tapi|tp)\s+(?:ribet|susah|rumit|berbelit|pusing)": "tidak mudah",
        r"(?:aman|secure|terpercaya)\s*(?:banget|bgt|sekali)?\s*(?:ya|sih|nih)?\s*(?:sampe|sampai|padahal|tapi)\s+(?:kena hack|dibobol|diretas|hilang|raib|lenyap)": "tidak aman",
        r"(?:transparan|jelas|terbuka)\s*(?:banget|bgt)?\s*(?:ya|sih|nih)?\s*(?:padahal|tapi|tp)\s+(?:sembunyi|diam|tutup mulut|ga ada kabar|gak ada info)": "tidak transparan",

        # === POLA KATA POSITIF + HASIL NEGATIF ===
        r"(?:untung|cuan|profit|gain)\s*(?:banget|bgt|amat|sekali)?\s*(?:ya|sih|nih|dong)?\s*(?:,|!)?\s*(?:malah|justru|eh|kok)\s+(?:rugi|buntung|minus|loss|jebol|amblas)": "sangat merugikan",
        r"(?:investasi|nabung|tabung)\s+(?:cerdas|pintar|smart)\s*(?:ya|nih|sih)?\s*(?:sampe|sampai|padahal|malah)\s+(?:rugi|kecewa|bangkrut|susah)": "investasi buruk",
        r"(?:terbaik|nomor satu|no\.?\s*1|top)\s*(?:ya|nih|sih|dong)?\s*(?:,|!)?\s*(?:dalam|soal|urusan)\s+(?:tipu|bohong|rugi|zonk|scam|menipu)": "sangat buruk",
        r"(?:sukses|berhasil|lancar)\s*(?:banget|bgt)?\s*(?:ya|sih|nih)?\s*(?:sampe|sampai|padahal)\s+(?:rugi|gagal|bangkrut|kecewa|tipu)": "gagal total",

        # === POLA SPESIFIK EMAS DIGITAL ===
        r"(?:emas|gold)\s*(?:digital)?\s*(?:terbaik|terpercaya|aman)\s*(?:ya|nih|sih)?\s*(?:sampe|sampai|padahal|tapi)\s+(?:hilang|raib|lenyap|dibobol|scam|tipu)": "penipuan mengecewakan",
        r"(?:spread|biaya|fee)\s*(?:kecil|murah|gratis)\s*(?:ya|nih|sih|katanya)?\s*(?:padahal|tapi|tp|malah|kok)\s+(?:mahal|besar|mencekik|gede|gila)": "biaya mahal",
        r"(?:harga|price)\s*(?:beli|jual)\s*(?:bagus|oke|mantap)\s*(?:ya|nih|sih)?\s*(?:padahal|tapi|tp)\s+(?:kemahalan|mahal|selisih|jauh|rugi)": "harga tidak menguntungkan",
        r"(?:likuiditas|cairkan|jual)\s*(?:gampang|mudah|cepat)\s*(?:katanya|ya|nih|sih)?\s*(?:padahal|tapi|eh|malah)\s+(?:susah|lama|ribet|ditolak|gagal)": "likuiditas buruk",
        r"(?:pegadaian|tokopedia|shopee|bukaemas|lakuemas|indogold)\s+(?:terpercaya|aman|bagus|mantap)\s*(?:ya|nih|sih)?\s*(?:sampe|sampai|padahal|tapi)\s+(?:tipu|rugi|kecewa|scam|bermasalah|error)": "platform bermasalah",

        # === POLA HIPERBOLA NEGATIF ===
        r"(?:luar biasa|luar\s+biasa)\s*(?:banget)?\s*(?:ya|nih|sih)?\s*(?:sampe|sampai)\s+(?:rugi|kecewa|tipu|bangkrut|kapok)": "sangat buruk",
        r"(?:super|sangat|amat)\s+(?:keren|bagus|mantap|hebat)\s*(?:ya|sih|nih)?\s*(?:sampe|sampai|padahal|malah)\s+(?:rugi|kecewa|tipu|zonk)": "sangat buruk",
        r"(?:wow|wah|woah)\s*(?:,|!)?\s*(?:bagus|keren|mantap|hebat)\s*(?:banget)?\s*(?:ya|nih|sih)?\s*(?:padahal|tapi)\s+(.+)": "tidak bagus",

        # === POLA SINDIRAN HALUS ===
        r"(?:pantas|patut|wajar)\s*(?:ya|sih|nih|dong)?\s*(?:,|!)?\s*(?:kalo|kalau|jika)\s+(?:tipu|bohong|rugi|zonk|scam|menipu|kabur)": "sangat mengecewakan",
        r"(?:emang|memang)\s+(?:gitu|begitu|kayak gitu)\s*(?:ya|sih|nih)?\s*(?:caranya|modusnya|sistemnya)?\s*(?:tipu|bohong|nipu|kecohin)": "penipuan mengecewakan",
        r"(?:untung|syukurlah|alhamdulillah)\s+(?:udah|sudah)\s+(?:tau|tahu|sadar|ngerti)\s+(?:dari awal|lebih awal|duluan)\s+(?:itu|ini)?\s*(?:tipu|bohong|scam|zonk|penipuan)": "penipuan mengecewakan",
        r"(?:selamat|congrats|congrat)\s*(?:ya|nih|sih|deh)?\s*(?:udah|sudah|telah)?\s*(?:berhasil|sukses)\s+(?:tipu|bohong|rugiin|kecohin|scam)": "penipuan mengecewakan",
        r"(?:hebat|keren|salut)\s*(?:banget|bgt)?\s*(?:ya|sih|nih)?\s*(?:bisa|bisa-bisanya|bisanya)\s+(?:tipu|bohong|kabur|nipu|hilang|raib)": "sangat mengecewakan",

        # === POLA EKSPEKTASI vs REALITA ===
        r"(?:katanya|konon|kabarnya)\s+(?:bagus|keren|aman|terpercaya|mantap)\s*(?:,|\.|\!)?\s*(?:nyatanya|ternyata|faktanya|realitanya)\s+(?:tipu|zonk|rugi|scam|bohong|kecewa)": "tidak sesuai ekspektasi",
        r"(?:ekspektasi|ekspektasinya)\s*(?:vs|versus|:)?\s*(?:realita|realitanya|kenyataannya)\s*(?:,|:)?\s*(?:rugi|kecewa|tipu|zonk|scam)": "tidak sesuai ekspektasi",
        r"(?:dijanjiin|dijanjikan|katanya)\s+(?:untung|cuan|profit|aman|bagus)\s*(?:,|\.|\!)?\s*(?:malah|justru|ternyata|nyatanya)\s+(?:rugi|kecewa|tipu|zonk|hilang)": "tidak sesuai ekspektasi",
    }

    NORMALISASI_DATA = {
        "satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5",
        "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9", "sepuluh": "10",
        "rp ": "rp", "rp.": "rp", "idr": "rp",
        "gold": "emas",
        "toko online": "toko online", "olshop": "toko online", "online shop": "toko online",
        "free ongkir": "gratis ongkos kirim",
        "cod": "bayar di tempat", "cash on delivery": "bayar di tempat",
        "dp": "uang muka", "downpayment": "uang muka",
        "ready stock": "tersedia", "ready": "tersedia",
        "pre order": "pemesanan dulu", "po": "pemesanan dulu",
        "sold out": "habis terjual", "habis": "habis terjual",
    }

    def normalisasi_emoji(self, teks):
        teks = str(teks)
        for pattern, ganti in self._re_emoji_ascii:
            teks = pattern.sub(ganti, teks)
        teks = self._re_repeat.sub(r'\1\1', teks)
        teks = self._re_exclaim.sub(' sangat ', teks)
        teks = self._re_question.sub(' tidak jelas ', teks)
        return self._re_ws.sub(' ', teks).strip()

    def normalisasi_sarkasme(self, teks):
        teks_lower = str(teks).lower()
        # ── OPTIMASI: compiled + early-exit saat pola pertama cocok ───────────
        for pattern, ganti in self._re_sarkasme:
            if pattern.search(teks_lower):
                return ganti + " " + teks_lower
        return teks

    def normalisasi_data(self, teks):
        teks = str(teks).lower()
        teks = self._re_repeat.sub(r'\1\1', teks)
        # ── OPTIMASI: compiled regex dari __init__ ─────────────────────────────
        for pattern, ganti in self._re_normdata:
            teks = pattern.sub(ganti, teks)
        return teks

    def normalisasi_slang(self, teks):
        teks = self.normalisasi_emoji(teks)
        teks = self.normalisasi_sarkasme(teks)
        teks = self.normalisasi_data(teks)
        # Selalu terapkan KAMUS_SLANG dulu agar kata gaul/singkatan terkonversi
        teks = " ".join([self.KAMUS_SLANG.get(k.lower(), k) for k in str(teks).split()])
        if NLP_ID_AVAILABLE and self.lemmatizer:
            teks = self.lemmatizer.lemmatize(teks)
        return teks

    def tokenisasi(self, teks):
        return str(teks).split()

    def hapus_stopword(self, token_list):
        teks_gabung = ' '.join(token_list)
        if NLP_ID_AVAILABLE and self.stopword_remover:
            teks_bersih = self.stopword_remover.remove_stopword(teks_gabung)
            return [k for k in teks_bersih.split() if len(k) > 2]
        stopwords_id = {
            'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan',
            'untuk', 'ada', 'adalah', 'pada', 'saya', 'kita', 'kami',
            'mereka', 'juga', 'akan', 'saat',
            'oleh', 'atau', 'jika', 'maka', 'bagi',
            'agar', 'apa', 'siapa', 'pun', 'pula',
            'bahwa', 'sehingga', 'ya', 'iya',
            # CATATAN: 'tidak', 'bisa', 'sudah', 'lebih', 'karena',
            # 'namun', 'tetapi', 'tapi', 'bagaimana', 'mau' sengaja TIDAK
            # dihapus karena mengandung nilai sentimen penting
        }
        return [k for k in token_list if k not in stopwords_id and len(k) > 2]

    def stemming(self, token_list):
        if SASTRAWI_AVAILABLE and self.stemmer:
            hasil = []
            for token in token_list:
                if token not in self._stem_cache:
                    # ── OPTIMASI: simpan ke cache, kata sama tidak di-stem ulang ──
                    self._stem_cache[token] = self.stemmer.stem(token)
                hasil.append(self._stem_cache[token])
            return hasil
        return token_list

    def proses_lengkap(self, teks):
        teks = self.konversi_emoji(teks)
        teks = self.bersihkan_teks(teks)
        teks = self.case_folding(teks)
        teks = self.normalisasi_slang(teks)
        token = self.tokenisasi(teks)
        token = self.hapus_stopword(token)
        token = self.stemming(token)
        return ' '.join(token)

    def proses_step(self, teks, step):
        hasil = {
            'teks_asli': teks,
            'konversi_emoji': '',
            'cleaning': '',
            'case_folding': '',
            'normalisasi': '',
            'tokenisasi': '',
            'stopword': '',
            'stemming': '',
        }
        t = self.konversi_emoji(teks)
        hasil['konversi_emoji'] = t
        if step == 'konversi_emoji': return hasil

        t = self.bersihkan_teks(t)
        hasil['cleaning'] = t
        if step == 'cleaning': return hasil

        t = self.case_folding(t)
        hasil['case_folding'] = t
        if step == 'case_folding': return hasil

        t = self.normalisasi_slang(t)
        hasil['normalisasi'] = t
        if step == 'normalisasi': return hasil

        tok = self.tokenisasi(t)
        hasil['tokenisasi'] = ' | '.join(tok)
        if step == 'tokenisasi': return hasil

        tok = self.hapus_stopword(tok)
        hasil['stopword'] = ' '.join(tok)
        if step == 'stopword': return hasil

        tok = self.stemming(tok)
        hasil['stemming'] = ' '.join(tok)
        return hasil


# Inisialisasi preprocessor global
preprocessor = PreprocessorTeks()

# =====================
# SMART CSV READER — auto-detect separator & kolom
# =====================
def read_csv_smart(path):
    """
    Baca CSV dengan deteksi otomatis. Mendukung berbagai format:
    - Separator koma (,) atau titik koma (;)
    - Format: @username,teks,;Label  (separator ; dengan koma di dalam teks)
    - Format: teks,label  atau  teks;label  (standar)
    - Kolom teks: 'teks', 'text', 'komentar', 'ulasan', 'review', 'content',
                  'tweet', 'Data', 'data', 'kalimat', 'sentence'
    - Kolom label: 'label', 'Label', 'Label Sentimen', 'sentimen', 'kategori'
    Selalu return DataFrame dengan kolom bernama 'teks' dan 'label' (jika ada).
    """
    LABEL_VALID = {'positif', 'negatif', 'netral', 'positive', 'negative',
                   'neutral', 'pos', 'neg', 'net'}
    PETA_LABEL = {
        'positif': 'positif', 'positive': 'positif', 'pos': 'positif',
        'negatif': 'negatif', 'negative': 'negatif', 'neg': 'negatif',
        'netral': 'netral', 'neutral': 'netral', 'net': 'netral',
        'bet': 'netral', 'nnet': 'netral', 'pneg': 'negatif',
        '[net': 'netral', 'nett': 'netral', 'poss': 'positif',
    }

    def bersihkan_label(x):
        x = str(x).strip().lower()
        return PETA_LABEL.get(x, x)

    def hapus_username(teks):
        teks = str(teks).strip()
        match = re.match(r'^@[\w.]+\s*,\s*(.*)', teks, re.DOTALL)
        if match:
            return match.group(1).strip()
        if re.match(r'^@[\w.]+\s*$', teks):
            return ''
        return teks

    # --- Baca semua baris mentah ---
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
        raw_lines = f.readlines()

    if not raw_lines:
        return pd.DataFrame(columns=['teks', 'label'])

    header_raw = raw_lines[0].strip()

    # -------------------------------------------------------
    # DETEKSI FORMAT KHUSUS: @username,teks,;Label
    # Ciri: header hanya berisi ";<nama_kolom_label>" tanpa kolom teks,
    # dan baris data diakhiri dengan ";Nilai_Label"
    # -------------------------------------------------------
    kandidat_label_nama = ['label sentimen', 'label', 'sentimen', 'kategori',
                           'class', 'kelas', 'sentiment']
    header_stripped = header_raw.lstrip(';').strip().lower()
    is_format_khusus = (
        header_raw.startswith(';') and
        header_stripped in kandidat_label_nama and
        len(raw_lines) > 1 and
        ';' in raw_lines[1]
    )

    if is_format_khusus:
        # Format: baris header = ";Label Sentimen"
        # Baris data = @username,teks komentar,;NilaiLabel
        # Strategi: ambil semua teks sebelum ';' terakhir sebagai kolom teks,
        #           ambil nilai setelah ';' terakhir sebagai label.
        records = []
        for line in raw_lines[1:]:
            line = line.strip()
            if not line:
                continue

            # Cari posisi ';' terakhir
            idx_semicolon = line.rfind(';')
            if idx_semicolon == -1:
                # Tidak ada ';' → tidak punya label, jadikan teks saja
                teks_raw = line
                label_raw = ''
            else:
                teks_raw  = line[:idx_semicolon].strip()
                label_raw = line[idx_semicolon + 1:].strip()

            # Handle tanda kutip ganda di awal/akhir (quoted CSV)
            if teks_raw.startswith('"') and teks_raw.endswith('"'):
                teks_raw = teks_raw[1:-1].replace('""', '"')

            # Buang trailing koma jika ada (misal: "teks,")
            teks_raw = teks_raw.rstrip(',').strip()

            # Buang @username di awal
            teks_bersih = hapus_username(teks_raw)

            records.append({'teks': teks_bersih, 'label': label_raw})

        df = pd.DataFrame(records)
        df['label'] = df['label'].apply(bersihkan_label)
        df = df[df['teks'].str.len() > 2].reset_index(drop=True)
        return df

    # -------------------------------------------------------
    # FORMAT STANDAR: deteksi separator otomatis
    # -------------------------------------------------------
    sep = ';' if header_raw.count(';') > header_raw.count(',') else ','
    df = pd.read_csv(path, sep=sep, encoding='utf-8-sig', on_bad_lines='skip')

    # Hapus kolom Unnamed
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    # Normalisasi nama kolom teks → 'teks'
    kandidat_teks = ['teks', 'text', 'komentar', 'ulasan', 'review',
                     'content', 'tweet', 'data', 'kalimat', 'sentence']
    for col in df.columns:
        if col.strip().lower() in kandidat_teks:
            if col != 'teks':
                df.rename(columns={col: 'teks'}, inplace=True)
            break

    # Normalisasi nama kolom label → 'label'
    for col in df.columns:
        if col.strip().lower() in kandidat_label_nama:
            if col != 'label':
                df.rename(columns={col: 'label'}, inplace=True)
            break

    # Normalisasi isi kolom label
    if 'label' in df.columns:
        df['label'] = df['label'].apply(bersihkan_label)

    # Bersihkan kolom teks: hapus @username di awal
    if 'teks' in df.columns:
        df['teks'] = df['teks'].apply(hapus_username)
        df = df[df['teks'].str.len() > 2].reset_index(drop=True)

    return df


# =====================
# AUTO DETECT KOLOM TEKS
# =====================
def detect_text_column(df):
    kandidat = ['teks', 'komentar', 'text', 'ulasan', 'review', 'content',
                'tweet', 'data', 'kalimat', 'sentence']
    for col in df.columns:
        if col.strip().lower() in kandidat:
            return col
    return df.columns[0]

# =====================
# USER MODEL
# =====================
class User(db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email    = db.Column(db.String(100))
    password = db.Column(db.String(255))

with app.app_context():
    db.create_all()

# =====================
# LOGIN & REGISTER
# =====================
@app.route('/')
def index():
    import json as _json
    # Baca statistik langsung — tidak perlu fetch JS terpisah
    pos = neg = net = total = 0
    akurasi = jumlah_train = jumlah_test = 0
    model_ready = os.path.exists(MODEL_PATH)

    try:
        preprocessing_path = os.path.join(UPLOAD_FOLDER, 'hasil_preprocessing.csv')
        csvs = [f for f in os.listdir(UPLOAD_FOLDER)
                if f.endswith('.csv') and f not in ('testing.csv',)]
        if os.path.exists(preprocessing_path):
            dataset_path = preprocessing_path
        elif csvs:
            dataset_path = os.path.join(UPLOAD_FOLDER, sorted(csvs)[0])
        else:
            dataset_path = None

        if dataset_path:
            df = read_csv_smart(dataset_path)
            total = len(df)
            if 'label' in df.columns:
                lbl = df['label'].astype(str).str.strip().str.lower()
                pos = int((lbl == 'positif').sum())
                neg = int((lbl == 'negatif').sum())
                net = int((lbl == 'netral').sum())
    except Exception:
        pass

    cm_data = cm_labels = report_data = None
    presisi_w = recall_w = f1_w = 0.0
    cv_acc_scores = cv_f1_scores = []
    cv_acc_mean = cv_acc_std = cv_f1_mean = 0.0

    result_path = os.path.join(MODEL_FOLDER, 'training_result.json')
    if os.path.exists(result_path):
        try:
            with open(result_path, encoding='utf-8') as _f:
                saved = _json.load(_f)
            akurasi       = round(float(saved.get('akurasi', 0) or 0), 2)
            jumlah_train  = int(saved.get('jumlah_train', 0) or 0)
            jumlah_test   = int(saved.get('jumlah_test', 0) or 0)
            cm_data       = saved.get('cm_data')
            cm_labels     = saved.get('cm_labels')
            report_data   = saved.get('report_data')
            presisi_w     = round(float(saved.get('presisi_w', 0) or 0), 2)
            recall_w      = round(float(saved.get('recall_w', 0) or 0), 2)
            f1_w          = round(float(saved.get('f1_w', 0) or 0), 2)
            cv_acc_scores = saved.get('cv_acc_scores', [])
            cv_f1_scores  = saved.get('cv_f1_scores', [])
            cv_acc_mean   = round(float(saved.get('cv_acc_mean', 0) or 0), 2)
            cv_acc_std    = round(float(saved.get('cv_acc_std', 0) or 0), 2)
            cv_f1_mean    = round(float(saved.get('cv_f1_mean', 0) or 0), 2)
        except Exception:
            pass

    import json as _json2
    return render_template("user.html",
        total_data    = total,
        positif       = pos,
        negatif       = neg,
        netral        = net,
        akurasi       = akurasi,
        jumlah_train  = jumlah_train,
        jumlah_test   = jumlah_test,
        model_ready   = model_ready,
        cm_data       = cm_data,
        cm_labels     = cm_labels,
        report_data   = report_data,
        presisi_w     = presisi_w,
        recall_w      = recall_w,
        f1_w          = f1_w,
        cv_acc_scores = _json2.dumps(cv_acc_scores),
        cv_f1_scores  = _json2.dumps(cv_f1_scores),
        cv_acc_mean   = cv_acc_mean,
        cv_acc_std    = cv_acc_std,
        cv_f1_mean    = cv_f1_mean,
    )

@app.route('/admin/login')
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login-proses', methods=['POST'])
def login_proses():
    user = User.query.filter_by(
        username=request.form['username'],
        password=request.form['password']
    ).first()
    if user:
        session['user'] = user.username
        return redirect(url_for('dashboard'))
    flash("Login gagal! Username atau password salah.")
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash("Username dan password wajib diisi!")
            return render_template('register.html')

        existing = User.query.filter_by(username=username).first()
        if existing:
            flash("Username sudah digunakan. Pilih username lain.")
            return render_template('register.html')

        db.session.add(User(username=username, email=email, password=password))
        db.session.commit()
        flash("Register berhasil! Silakan login.")
        return redirect(url_for('login'))
    return render_template('register.html')

# =====================
# DASHBOARD
# =====================
@app.route('/dashboard')
@login_required
def dashboard():
    training = testing = pos = neg = net = akurasi = 0
    nama_file = None

    active_dataset = get_dataset_file()
    if active_dataset and os.path.exists(active_dataset):
        nama_file = os.path.basename(active_dataset)
        df        = read_csv_smart(active_dataset)
        total     = len(df)
        # Data training = 80% dari total (sesuai rasio default klasifikasi)
        training  = int(total * 0.8)
        if 'label' in df.columns:
            df['label'] = df['label'].astype(str).str.strip().str.lower()
            pos = len(df[df['label'] == 'positif'])
            neg = len(df[df['label'] == 'negatif'])
            net = len(df[df['label'] == 'netral'])

    # Akurasi & testing diisi dari session setelah training, atau dari file hasil training
    import json as _json
    result_path = os.path.join(MODEL_FOLDER, 'training_result.json')
    if os.path.exists(result_path):
        with open(result_path, encoding='utf-8') as _f:
            saved = _json.load(_f)
        akurasi = saved.get('akurasi', session.get('akurasi_model', 0))
        testing = saved.get('jumlah_test', session.get('jumlah_testing', 0))
        training = saved.get('jumlah_train', training)
    else:
        akurasi = session.get('akurasi_model', 0)
        testing = session.get('jumlah_testing', 0)

    # ── KEYWORD EXTRACTION (Top 20) ──────────────────────────────────────────
    import re as _re
    from collections import Counter as _Counter
    top_keywords = []
    try:
        from nltk.corpus import stopwords as _sw
        import nltk as _nltk
        _nltk.download('stopwords', quiet=True)
        _stop_en = set(_sw.words('english'))
    except Exception:
        _stop_en = set()

    _stop_id = {
        'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
        'ada', 'tidak', 'ya', 'juga', 'saya', 'aku', 'kamu', 'dia', 'mereka',
        'kita', 'kami', 'adalah', 'atau', 'jika', 'kalau', 'pada', 'karena',
        'tapi', 'bisa', 'sudah', 'akan', 'lebih', 'sangat', 'paling',
        'banyak', 'saja', 'lagi', 'dong', 'deh', 'sih', 'nih', 'kan', 'udah',
        'nya', 'kak', 'bang', 'min', 'mas', 'pak', 'bu', 'ga', 'gak', 'nggak',
        'mau', 'buat', 'aja', 'yg', 'jd', 'lg', 'tp', 'dtg', 'dr', 'utk',
        'gimana', 'apakah', 'bagaimana', 'berapa', 'siapa', 'apa',
        'tolong', 'mending', 'saat', 'hal', 'cara', 'bagi',
        'dlm', 'krn', 'dgn', 'yaa', 'wah', 'haha', 'hehe', 'loh', 'dah', 'nah',
    }
    _all_sw = _stop_en | _stop_id

    def _kw_preprocess(text):
        text = str(text).lower()
        text = _re.sub(r'@\w+', '', text)
        text = _re.sub(r'http\S+|www\.\S+', '', text)
        text = _re.sub(r'[^a-z\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if t not in _all_sw and len(t) > 2]

    if active_dataset and os.path.exists(active_dataset):
        try:
            _df_kw = read_csv_smart(active_dataset)
            _col = None
            for _c in ['Data', 'teks_asli', 'teks', 'text', 'content', 'komentar']:
                if _c in _df_kw.columns:
                    _col = _c
                    break
            if _col is None:
                _col = _df_kw.columns[0]
            _all_tokens = []
            for _t in _df_kw[_col].dropna():
                _all_tokens.extend(_kw_preprocess(str(_t)))
            _freq = _Counter(_all_tokens)
            top_keywords = [{'word': w, 'count': c} for w, c in _freq.most_common(20)]
        except Exception:
            top_keywords = []

    return render_template(
        "dashboard.html",
        training=training, testing=testing, akurasi=akurasi,
        positif=pos, negatif=neg, netral=net,
        nama_file=nama_file,
        top_keywords=top_keywords,
    )

# =====================
# DATASET
# =====================
@app.route("/dataset", methods=["GET", "POST"])
@login_required
def dataset():
    if request.method == "POST":
        file = request.files.get("file_csv")
        if file and file.filename.endswith(".csv"):
            filename = secure_filename(file.filename)
            path     = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            flash(f"Upload berhasil: {filename}")
        else:
            flash("File harus berformat CSV!")

    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".csv")]
    data  = []
    for i, f in enumerate(files, 1):
        file_path = os.path.join(UPLOAD_FOLDER, f)
        try:
            df     = read_csv_smart(file_path)
            jumlah = len(df) if not df.empty else 0
            status = "OK"
        except:
            jumlah = 0
            status = "ERROR"
        data.append({"id": i, "filename": f, "jumlah": jumlah, "status": status})

    return render_template("dataset.html", datasets=data)

# =====================
# DETAIL DATASET (AJAX)
# =====================
@app.route("/detail-data/<path:filename>")
@login_required
def detail_data(filename):
    filename = unquote(filename)
    path     = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File tidak ditemukan"})
    df         = read_csv_smart(path)
    page       = request.args.get("page", 1, type=int)
    per_page   = request.args.get("per_page", 10, type=int)
    total_data = len(df)
    total_pages = math.ceil(total_data / per_page)
    start      = (page - 1) * per_page
    end        = start + per_page
    data       = df.iloc[start:end].fillna("").to_dict(orient="records")
    return jsonify({
        "columns": list(df.columns), "data": data,
        "page": page, "per_page": per_page,
        "total_pages": total_pages, "total_data": total_data
    })

# =====================
# DELETE DATASET
# =====================
@app.route('/delete/<filename>')
@login_required
def delete(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        flash(f"File '{filename}' berhasil dihapus!")
    else:
        flash("File tidak ditemukan!")
    return redirect(url_for('dataset'))

# =====================
# PILIH DATASET & REDIRECT KE PREPROCESSING
# =====================
@app.route('/clean/<filename>')
@login_required
def clean(filename):
    global temp_df, current_file
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        flash("File tidak ditemukan!")
        return redirect(url_for('dataset'))
    df   = read_csv_smart(path)
    temp_df      = df.copy()
    current_file = filename
    session['selected_file'] = filename

    # Hapus hasil preprocessing & training lama agar tidak muncul dari dataset sebelumnya
    for _old_file in ['hasil_preprocessing.csv']:
        _old_path = os.path.join(UPLOAD_FOLDER, _old_file)
        if os.path.exists(_old_path):
            os.remove(_old_path)
    session.pop('hasil_preprocessing_ready', None)

    flash(f"Dataset '{filename}' dipilih. Lakukan preprocessing step by step.")
    return redirect(url_for('hasil'))

# =====================
# HASIL PREPROCESSING
# =====================
@app.route('/hasil')
@login_required
def hasil():
    preprocessing_path = os.path.join(UPLOAD_FOLDER, 'hasil_preprocessing.csv')

    # Kalau hasil preprocessing sudah ada di disk, langsung tampilkan dari sana
    if os.path.exists(preprocessing_path):
        df = pd.read_csv(preprocessing_path, encoding='utf-8-sig')
        return render_template('hasil_processing.html',
                               data=df.head(10).to_dict('records'),
                               lib_status=_get_lib_status(),
                               hasil_ready=True)

    # Belum ada hasil → tampilkan form seperti biasa
    df_tmp = _get_temp_df()
    if df_tmp is None:
        return render_template('hasil_processing.html', data=None,
                               lib_status=_get_lib_status(),
                               hasil_ready=False)
    df    = df_tmp.copy()
    kolom = detect_text_column(df)
    df["teks_asli"] = df[kolom]
    return render_template('hasil_processing.html',
                           data=df.head(10).to_dict('records'),
                           lib_status=_get_lib_status(),
                           hasil_ready=False)

def _get_lib_status():
    return {
        'emoji': EMOJI_AVAILABLE,
        'nlp_id': NLP_ID_AVAILABLE,
        'sastrawi': SASTRAWI_AVAILABLE,
    }

# =====================
# API STEP PREPROCESSING (per-step, AJAX)
# =====================
@app.route('/step/<step>')
@login_required
def step(step):
    df_tmp = _get_temp_df()

    if df_tmp is None:
        return jsonify({"error": "Dataset belum dipilih. Silakan pilih dataset terlebih dahulu."})

    VALID_STEPS = ['konversi_emoji', 'cleaning', 'case_folding', 'normalisasi',
                   'tokenisasi', 'stopword', 'stemming']

    if step not in VALID_STEPS:
        return jsonify({"error": f"Step '{step}' tidak dikenal."})

    df    = df_tmp.copy()
    kolom = detect_text_column(df)

    hasil_rows = []
    for _, row in df.head(10).iterrows():
        teks = str(row[kolom])
        r = preprocessor.proses_step(teks, step)
        hasil_rows.append(r)

    return jsonify(hasil_rows)

# =====================
# PROSES SEMUA & SIMPAN (AJAX) — alias, diarahkan ke proses_semua_preprocessing
# =====================
@app.route('/proses-semua', methods=['POST'])
@login_required
def proses_semua():
    return proses_semua_preprocessing()

# =====================
# PROSES SEMUA & SIMPAN — pakai label asli dari dataset
# =====================
@app.route('/proses-semua-preprocessing', methods=['POST'])
@login_required
def proses_semua_preprocessing():
    try:
        df_tmp = _get_temp_df()
        if df_tmp is None:
            return jsonify({"error": "Dataset belum dipilih."})

        df          = df_tmp.copy()
        kolom_teks  = detect_text_column(df)
        punya_label = 'label' in df.columns

        # ── OPTIMASI: vectorized apply — jauh lebih cepat dari iterrows ────────
        seri = df[kolom_teks].astype(str)

        col_emoji   = seri.apply(preprocessor.konversi_emoji)
        col_clean   = col_emoji.apply(preprocessor.bersihkan_teks)
        col_case    = col_clean.str.lower()
        col_norm    = col_case.apply(preprocessor.normalisasi_slang)
        col_token   = col_norm.apply(preprocessor.tokenisasi)
        col_stop    = col_token.apply(preprocessor.hapus_stopword)
        col_stem    = col_stop.apply(preprocessor.stemming)

        hasil_df = pd.DataFrame({
            'teks_asli'     : seri,
            'konversi_emoji': col_emoji,
            'cleaning'      : col_clean,
            'case_folding'  : col_case,
            'normalisasi'   : col_norm,
            'tokenisasi'    : col_token.apply(lambda t: ' | '.join(t)),
            'stopword'      : col_stop.apply(lambda t: ' '.join(t)),
            'stemming'      : col_stem.apply(lambda t: ' '.join(t)),
            'teks_bersih'   : col_stem.apply(lambda t: ' '.join(t)),
            'label'         : (df['label'].astype(str).str.strip().str.lower().values
                               if punya_label else 'netral'),
        })

        output_path = os.path.join(UPLOAD_FOLDER, 'hasil_preprocessing.csv')
        hasil_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        session['hasil_preprocessing_ready'] = True
        preview = hasil_df.fillna('').to_dict(orient='records')

        return jsonify({
            "success": True,
            "jumlah" : len(hasil_df),
            "preview": preview,
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# =====================
# DOWNLOAD HASIL PREPROCESSING
# =====================
@app.route('/download-hasil')
@login_required
def download_hasil():
    output_path = os.path.join(UPLOAD_FOLDER, 'hasil_preprocessing.csv')
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True,
                         download_name='hasil_preprocessing.csv')
    flash("Belum ada hasil preprocessing. Silakan proses data terlebih dahulu.")
    return redirect(url_for('hasil'))

# =====================
# API PAGINATION HASIL PREPROCESSING
# =====================
@app.route('/hasil-data')
@login_required
def hasil_data():
    output_path = os.path.join(UPLOAD_FOLDER, 'hasil_preprocessing.csv')
    if not os.path.exists(output_path):
        return jsonify({"error": "Belum ada hasil preprocessing."})
    df       = pd.read_csv(output_path, encoding='utf-8-sig')
    page     = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    total    = len(df)
    start    = (page - 1) * per_page
    data     = df.iloc[start:start + per_page].fillna('').to_dict(orient='records')
    return jsonify({
        "data"       : data,
        "total"      : total,
        "page"       : page,
        "per_page"   : per_page,
        "total_pages": math.ceil(total / per_page),
    })

# =====================
# KLASIFIKASI (TRAINING NAIVE BAYES)
# =====================
@app.route('/klasifikasi', methods=['GET', 'POST'])
@login_required
def klasifikasi():
    akurasi      = session.get('akurasi_model', 0)
    cm_data      = None
    cm_labels    = None
    report_data  = None
    jumlah_train = 0
    jumlah_test  = 0
    positif = negatif = netral = 0
    total_data   = 0

    selected_file = session.get('selected_file')

    # Prioritaskan hasil preprocessing jika ada
    preprocessing_path = os.path.join(UPLOAD_FOLDER, 'hasil_preprocessing.csv')
    dataset_path = None
    dataset_source = None

    if os.path.exists(preprocessing_path):
        dataset_path   = preprocessing_path
        dataset_source = 'hasil_preprocessing.csv'
    elif selected_file:
        candidate = os.path.join(UPLOAD_FOLDER, selected_file)
        if os.path.exists(candidate):
            dataset_path   = candidate
            dataset_source = selected_file

    if not dataset_path:
        flash("Belum ada data preprocessing. Lakukan preprocessing terlebih dahulu!")
        return render_template('klasifikasi.html', akurasi=0,
                               positif=0, negatif=0, netral=0,
                               total_data=0, jumlah_train=0, jumlah_test=0,
                               dataset_source=None, cm_data=None, cm_labels=None,
                               report_data=None)

    df_count = read_csv_smart(dataset_path)
    total_data = len(df_count)
    if 'label' in df_count.columns:
        df_count['label'] = df_count['label'].astype(str).str.lower()
        positif = len(df_count[df_count['label'] == 'positif'])
        negatif = len(df_count[df_count['label'] == 'negatif'])
        netral  = len(df_count[df_count['label'] == 'netral'])

    if request.method == 'POST':
        try:
            df = read_csv_smart(dataset_path)

            if 'label' not in df.columns:
                flash("Kolom 'label' tidak ditemukan! Lakukan preprocessing terlebih dahulu.")
                return render_template('klasifikasi.html', akurasi=0,
                                       positif=positif, negatif=negatif, netral=netral,
                                       total_data=total_data, jumlah_train=0, jumlah_test=0,
                                       dataset_source=dataset_source, cm_data=None, cm_labels=None,
                                       report_data=None)

            X = df['teks_bersih'].astype(str) if 'teks_bersih' in df.columns \
                else df[detect_text_column(df)].apply(preprocessor.proses_lengkap)

            y = df['label'].astype(str).str.strip()

            # === FIX BUG: Buang baris dengan label tidak valid (nan, kosong, dll) ===
            label_valid = {'positif', 'negatif', 'netral'}
            mask_label = y.isin(label_valid)
            X = X[mask_label].reset_index(drop=True)
            y = y[mask_label].reset_index(drop=True)

            # === FIX BUG: Buang baris teks_bersih yang kosong (cegah empty vocabulary) ===
            mask_nonempty = X.str.strip().str.len() > 0
            X = X[mask_nonempty].reset_index(drop=True)
            y = y[mask_nonempty].reset_index(drop=True)

            # === Validasi jumlah data minimum ===
            if len(X) < 10:
                flash(f"Data valid terlalu sedikit ({len(X)} baris). "
                      f"Minimal 10 data berlabel positif/negatif/netral untuk training.")
                return render_template('klasifikasi.html', akurasi=0,
                                       positif=positif, negatif=negatif, netral=netral,
                                       total_data=total_data, jumlah_train=0, jumlah_test=0,
                                       dataset_source=dataset_source, cm_data=None, cm_labels=None,
                                       report_data=None)

            rasio = float(request.form.get('rasio', 0.2))

            # === FIX BUG: Tambah stratify agar distribusi kelas proporsional ===
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=rasio, random_state=42, stratify=y
            )
            jumlah_train = len(X_train)
            jumlah_test  = len(X_test)

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
                ('nb', MultinomialNB(alpha=1.0))
            ])
            pipeline.fit(X_train, y_train)

            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(pipeline, f)

            # Reset cache agar model terbaru langsung dipakai
            global _pipeline_cache
            _pipeline_cache = None

            y_pred  = pipeline.predict(X_test)
            akurasi = round(accuracy_score(y_test, y_pred) * 100, 2)

            session['akurasi_model']  = akurasi
            session['jumlah_testing'] = jumlah_test
            session['jumlah_train']   = jumlah_train
            session['jumlah_test']    = jumlah_test

            labels    = sorted(list(y.unique()))
            cm        = confusion_matrix(y_test, y_pred, labels=labels)
            cm_data   = list(enumerate(cm.tolist()))
            cm_labels = labels

            # Classification report per label
            report_dict = classification_report(y_test, y_pred, labels=labels,
                                                output_dict=True, zero_division=0)
            report_data = []
            for lbl in labels:
                if lbl in report_dict:
                    r = report_dict[lbl]
                    report_data.append({
                        'label'    : lbl,
                        'precision': round(r['precision'] * 100, 2),
                        'recall'   : round(r['recall'] * 100, 2),
                        'f1'       : round(r['f1-score'] * 100, 2),
                        'support'  : int(r['support']),
                    })

            # === 5-FOLD STRATIFIED CROSS VALIDATION ===
            cv_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
                ('nb', MultinomialNB(alpha=1.0))
            ])
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = cross_validate(
                cv_pipeline, X, y, cv=skf,
                scoring=['accuracy', 'f1_weighted'],
                return_train_score=False
            )
            cv_acc_scores  = [round(v * 100, 2) for v in cv_results['test_accuracy'].tolist()]
            cv_f1_scores   = [round(v * 100, 2) for v in cv_results['test_f1_weighted'].tolist()]
            cv_acc_mean    = round(float(sum(cv_acc_scores) / len(cv_acc_scores)), 2)
            cv_acc_std     = round(float((sum((x - cv_acc_mean)**2 for x in cv_acc_scores) / len(cv_acc_scores)) ** 0.5), 2)
            cv_f1_mean     = round(float(sum(cv_f1_scores) / len(cv_f1_scores)), 2)

            # Weighted metrics
            from sklearn.metrics import precision_score, recall_score
            presisi_w = round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
            recall_w  = round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
            f1_w      = round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)

            flash(f"Training selesai! Akurasi: {akurasi}%")

            # Simpan hasil training ke file agar tetap ada saat halaman dibuka ulang
            import json as _json
            training_result = {
                'cm_data'       : cm_data,
                'cm_labels'     : cm_labels,
                'report_data'   : report_data,
                'jumlah_train'  : jumlah_train,
                'jumlah_test'   : jumlah_test,
                'akurasi'       : akurasi,
                'presisi_w'     : presisi_w,
                'recall_w'      : recall_w,
                'f1_w'          : f1_w,
                'cv_acc_scores' : cv_acc_scores,
                'cv_f1_scores'  : cv_f1_scores,
                'cv_acc_mean'   : cv_acc_mean,
                'cv_acc_std'    : cv_acc_std,
                'cv_f1_mean'    : cv_f1_mean,
            }
            result_path = os.path.join(MODEL_FOLDER, 'training_result.json')
            with open(result_path, 'w', encoding='utf-8') as _f:
                _json.dump(training_result, _f, ensure_ascii=False)

        except ValueError as ve:
            flash(f"Gagal training: {str(ve)}")
        except Exception as e:
            flash(f"Terjadi kesalahan tidak terduga saat training: {str(e)}")

    else:
        result_path = os.path.join(MODEL_FOLDER, 'training_result.json')
        if os.path.exists(result_path):
            import json as _json
            with open(result_path, encoding='utf-8') as _f:
                saved = _json.load(_f)
            cm_data      = saved.get('cm_data')
            cm_labels    = saved.get('cm_labels')
            report_data  = saved.get('report_data')
            jumlah_train = saved.get('jumlah_train', 0)
            jumlah_test  = saved.get('jumlah_test', 0)
            akurasi      = saved.get('akurasi', akurasi)
        else:
            jumlah_train = session.get('jumlah_train', 0)
            jumlah_test  = session.get('jumlah_test', 0)

    return render_template(
        'klasifikasi.html',
        akurasi      = akurasi,
        cm_data      = cm_data,
        cm_labels    = cm_labels,
        report_data  = report_data,
        positif      = positif,
        negatif      = negatif,
        netral       = netral,
        total_data   = total_data,
        jumlah_train = jumlah_train,
        jumlah_test  = jumlah_test,
        dataset_source = dataset_source,
    )

# =====================
# DATA TESTING
# =====================
@app.route('/datatesting', methods=['GET', 'POST'])
@login_required
def datatesting():
    hasil   = []
    ringkasan = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("Pilih file CSV terlebih dahulu!")
            return render_template('datatesting.html', hasil=hasil, ringkasan=None)
        if not file.filename.endswith('.csv'):
            flash("File harus berformat CSV!")
            return render_template('datatesting.html', hasil=hasil, ringkasan=None)
        if not os.path.exists(MODEL_PATH):
            flash("Model belum ditraining! Silakan ke halaman Klasifikasi dahulu.")
            return render_template('datatesting.html', hasil=hasil, ringkasan=None)

        path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(path)

        df    = read_csv_smart(path)
        kolom = detect_text_column(df)

        pipeline = _get_pipeline()
        if pipeline is None:
            flash("Model belum ditraining! Silakan ke halaman Klasifikasi dahulu.")
            return render_template('datatesting.html', hasil=hasil, ringkasan=None)

        teks = df[kolom].apply(preprocessor.proses_lengkap)
        df['prediksi'] = pipeline.predict(teks)

        total    = len(df)
        positif  = int((df['prediksi'] == 'positif').sum())
        negatif  = int((df['prediksi'] == 'negatif').sum())
        netral   = int((df['prediksi'] == 'netral').sum())
        ringkasan = {
            'total'  : total,
            'positif': positif,
            'negatif': negatif,
            'netral' : netral,
            'pct_pos': round(positif / total * 100, 1) if total else 0,
            'pct_neg': round(negatif / total * 100, 1) if total else 0,
            'pct_net': round(netral  / total * 100, 1) if total else 0,
        }

        session['jumlah_testing'] = total
        hasil = df.head(50).to_dict('records')

    return render_template('datatesting.html', hasil=hasil, ringkasan=ringkasan)

# =====================
# RULE-BASED SENTIMENT OVERRIDE
# =====================
def rule_based_sentimen(teks):
    """
    Cek kalimat dengan aturan kata kunci SEBELUM model prediksi.
    Return 'positif' / 'negatif' / None (None = biarkan model yg putuskan).
    Sistem skoring: hitung bobot positif vs negatif, ambil yang dominan.
    """
    t = teks.lower()

    # --- Kata kunci POSITIF dengan bobot ---
    KATA_POSITIF = {
        # Aksi beli / investasi emas (bobot tinggi)
        "beli emas": 3, "membeli emas": 3, "bli emas": 3,
        "beli emas digital": 4, "beli emas fisik": 4,
        "nabung emas": 3, "menabung emas": 3, "tabung emas": 3,
        "invest emas": 3, "investasi emas": 3, "inves emas": 3,
        "mulai invest": 2, "mulai beli": 2, "mulai nabung": 2,
        "tambah emas": 3, "tambah koleksi": 2,
        "cicil emas": 2, "nyicil emas": 2,
        "dca emas": 3, "beli bertahap": 2,
        "koleksi emas": 2, "punya emas": 2,
        "simpan emas": 2, "nyimpen emas": 2,

        # Harga naik (bobot tinggi)
        "harga emas naik": 4, "emas naik": 3, "harga naik": 3,
        "harga emas meroket": 4, "emas meroket": 4,
        "harga emas melonjak": 4, "emas melonjak": 4,
        "harga emas melambung": 4, "emas melambung": 4,
        "harga emas menguat": 3, "emas menguat": 3,
        "harga emas tinggi": 3, "emas makin tinggi": 3,
        "emas terus naik": 4, "emas lagi naik": 3,
        "naik terus": 3, "terus naik": 3,
        "all time high": 4, "ath": 3, "rekor tertinggi": 4,
        "bullish": 3, "rally": 3, "up terus": 3,
        "to the moon": 3, "moon": 2,

        # Keuntungan / profit
        "cuan": 3, "cuann": 3, "cuannn": 4,
        "untung": 2, "untungg": 3, "untunggg": 4,
        "profit": 3, "gain": 2,
        "balik modal": 3, "sudah balik modal": 4,
        "untung banyak": 4, "untung besar": 4, "untung gede": 4,
        "cuan banyak": 4, "cuan besar": 4,
        "hasil bagus": 3, "hasil memuaskan": 3,
        "imbal hasil bagus": 3, "return bagus": 3,

        # Ekspresi positif umum
        "bagus banget": 3, "bagus bgt": 3, "sangat bagus": 3,
        "mantap": 2, "mantap banget": 3, "mantap jiwa": 3,
        "jos": 2, "joss": 3, "josss": 4,
        "keren banget": 3, "keren bgt": 3,
        "worth it": 3, "worthit": 3, "worth banget": 4,
        "puas banget": 3, "puas bgt": 3, "sangat puas": 3,
        "senang banget": 3, "seneng banget": 3, "bahagia": 2,
        "recommended": 2, "rekomendasikan": 2, "rekomendasi": 2,
        "terbaik": 3, "the best": 3, "top banget": 3,
        "luar biasa": 2, "luar biasa banget": 3,
        "memuaskan": 2, "sangat memuaskan": 3,
        "aman banget": 3, "sangat aman": 3,
        "terpercaya banget": 3, "sangat terpercaya": 3,
        "no 1": 2, "nomor 1": 2, "nomor satu": 2,
        "happy": 2, "happy banget": 3,
        "seneng": 2, "senang": 2,
        "suka banget": 3, "suka bgt": 3,
        "cocok banget": 3, "cocok bgt": 3,
        "tepat banget": 3,
        "bijak": 2, "cerdas": 2,
        "peluang bagus": 3, "peluang emas": 3,
        "waktu tepat": 2, "saat tepat": 2,
        "investasi terbaik": 4, "investasi bagus": 3, "investasi cerdas": 3,
        "investasi aman": 3, "aset aman": 3,
        "lindung nilai": 2, "tahan inflasi": 3, "anti inflasi": 3,
        "stabil": 1, "nilainya stabil": 2, "harganya stabil": 2,
        "safe haven": 3,
        "ijo": 2, "hijau": 2,
        "naik": 2, "naek": 2, "meroket": 3, "melonjak": 3,
        "meningkat": 2, "menguat": 2,
        "apresiasi": 2, "terapresiasi": 2,
    }

    # --- Kata kunci NEGATIF dengan bobot ---
    KATA_NEGATIF = {
        # Harga turun
        "harga emas turun": 4, "emas turun": 3, "harga turun": 3,
        "harga emas anjlok": 4, "emas anjlok": 4,
        "harga emas ambles": 4, "emas ambles": 4,
        "harga emas jeblok": 4, "emas jeblok": 4,
        "harga emas merosot": 4, "emas merosot": 4,
        "harga emas longsor": 4, "emas longsor": 4,
        "harga emas crash": 4, "crash": 3,
        "bearish": 3, "downtrend": 3,
        "turun terus": 4, "terus turun": 4,
        "emas lagi turun": 3, "emas terus turun": 4,
        "anjlok": 3, "ambles": 3, "merosot": 3, "jeblok": 3,
        "turun": 2, "turunn": 3, "turunnnn": 4,
        "melemah": 2, "pelemahan": 2,

        # Kerugian
        "rugi": 2, "rugii": 3, "rugiii": 4,
        "rugi besar": 4, "rugi banyak": 4, "rugi gede": 4,
        "boncos": 3, "buntung": 3,
        "loss": 2, "capital loss": 3, "kapital loss": 3,
        "nyangkut": 3, "nyangkutt": 4,
        "minus": 2, "minuss": 3,
        "tidak untung": 3, "ga untung": 3, "gak untung": 3,
        "tidak balik modal": 3, "ga balik modal": 3,

        # Ekspresi negatif
        "kecewa banget": 3, "kecewa bgt": 3, "sangat kecewa": 3,
        "mengecewakan": 2, "sangat mengecewakan": 3,
        "parah banget": 3, "parah bgt": 3,
        "buruk banget": 3, "sangat buruk": 3,
        "jelek banget": 3, "sangat jelek": 3,
        "tidak puas": 3, "ga puas": 3, "gak puas": 3,
        "nyesel": 2, "menyesal": 2, "nyesel beli": 3,
        "kapok": 2, "kapok banget": 3, "kapok invest": 3,
        "ga mau lagi": 3, "gak mau lagi": 3, "tidak mau lagi": 3,
        "jangan beli": 3, "jangan invest": 3, "hindari": 2,
        "waspada": 1, "hati hati": 1, "berbahaya": 2,
        "bahaya banget": 3, "berisiko tinggi": 3,
        "zonk": 2, "zonk banget": 3,
        "scam": 3, "penipuan": 3, "ditipu": 3, "tertipu": 3,
        "tidak aman": 3, "ga aman": 3, "gak aman": 3,
        "tidak worth": 3, "ga worth": 3, "gak worth": 3,
        "tidak recommended": 3, "ga recommended": 3,
        "merah": 2,
        "dump": 2, "dumping": 2,
        "susah jual": 2, "ga bisa jual": 3, "susah cair": 2,
        "aplikasi error": 2, "sering error": 2, "sering down": 2,
        "cs ga respon": 2, "tidak ada respon": 2,
    }

    # --- Negasi yang membalik sentimen ---
    NEGASI = ["tidak", "ga", "gak", "bukan", "jangan", "belum", "tak",
              "tanpa", "anti", "kurang", "nggak", "ngga", "enggak"]

    skor_pos = 0
    skor_neg = 0

    # Cek frasa multi-kata dulu (prioritas lebih panjang)
    for frasa in sorted(KATA_POSITIF, key=len, reverse=True):
        if frasa in t:
            # Cek apakah ada negasi sebelum frasa ini
            idx = t.find(frasa)
            konteks = t[max(0, idx-20):idx]
            if any(neg in konteks.split() for neg in NEGASI):
                skor_neg += KATA_POSITIF[frasa]
            else:
                skor_pos += KATA_POSITIF[frasa]

    for frasa in sorted(KATA_NEGATIF, key=len, reverse=True):
        if frasa in t:
            idx = t.find(frasa)
            konteks = t[max(0, idx-20):idx]
            if any(neg in konteks.split() for neg in NEGASI):
                skor_pos += KATA_NEGATIF[frasa]
            else:
                skor_neg += KATA_NEGATIF[frasa]

    # Ambil keputusan jika skor cukup dominan
    selisih = skor_pos - skor_neg
    if selisih >= 2:
        return 'positif'
    elif selisih <= -2:
        return 'negatif'
    return None  # biarkan model yang putuskan


# =====================
# CEK KALIMAT (USER - publik, tanpa login)
# =====================
@app.route('/cek', methods=['GET', 'POST'])
def cek_user():
    import json as _json
    hasil         = None
    kalimat_input = None
    error_msg     = None

    if request.method == 'POST':
        teks = request.form.get('kalimat', '').strip()
        if not teks:
            error_msg = "Masukkan kalimat terlebih dahulu!"
        elif not os.path.exists(MODEL_PATH):
            error_msg = "Model belum ditraining. Silakan hubungi admin."
        else:
            pipeline = _get_pipeline()
            if pipeline is None:
                error_msg = "Model belum ditraining. Silakan hubungi admin."
            else:
                hasil = rule_based_sentimen(teks)
                if hasil is None:
                    clean = preprocessor.proses_lengkap(teks)
                    hasil = pipeline.predict([clean])[0]
                kalimat_input = teks
                session['cek_hasil_terakhir']   = hasil
                session['cek_kalimat_terakhir'] = kalimat_input
    else:
        hasil         = session.get('cek_hasil_terakhir')
        kalimat_input = session.get('cek_kalimat_terakhir')

    # Load stats sama seperti index()
    pos = neg = net = total = 0
    akurasi = jumlah_train = jumlah_test = 0
    model_ready = os.path.exists(MODEL_PATH)
    cm_data = cm_labels = report_data = None
    presisi_w = recall_w = f1_w = 0.0
    cv_acc_scores = cv_f1_scores = []
    cv_acc_mean = cv_acc_std = cv_f1_mean = 0.0

    try:
        preprocessing_path = os.path.join(UPLOAD_FOLDER, 'hasil_preprocessing.csv')
        dataset_path = preprocessing_path if os.path.exists(preprocessing_path) else get_dataset_file()
        if dataset_path and os.path.exists(dataset_path):
            df = read_csv_smart(dataset_path)
            total = len(df)
            if 'label' in df.columns:
                lbl = df['label'].astype(str).str.strip().str.lower()
                pos = int((lbl == 'positif').sum())
                neg = int((lbl == 'negatif').sum())
                net = int((lbl == 'netral').sum())
    except Exception:
        pass

    result_path = os.path.join(MODEL_FOLDER, 'training_result.json')
    if os.path.exists(result_path):
        try:
            with open(result_path, encoding='utf-8') as _f:
                saved = _json.load(_f)
            akurasi       = round(float(saved.get('akurasi', 0) or 0), 2)
            jumlah_train  = int(saved.get('jumlah_train', 0) or 0)
            jumlah_test   = int(saved.get('jumlah_test', 0) or 0)
            cm_data       = saved.get('cm_data')
            cm_labels     = saved.get('cm_labels')
            report_data   = saved.get('report_data')
            presisi_w     = round(float(saved.get('presisi_w', 0) or 0), 2)
            recall_w      = round(float(saved.get('recall_w', 0) or 0), 2)
            f1_w          = round(float(saved.get('f1_w', 0) or 0), 2)
            cv_acc_scores = saved.get('cv_acc_scores', [])
            cv_f1_scores  = saved.get('cv_f1_scores', [])
            cv_acc_mean   = round(float(saved.get('cv_acc_mean', 0) or 0), 2)
            cv_acc_std    = round(float(saved.get('cv_acc_std', 0) or 0), 2)
            cv_f1_mean    = round(float(saved.get('cv_f1_mean', 0) or 0), 2)
        except Exception:
            pass

    import json as _json2
    return render_template('user.html',
        cek_hasil         = hasil,
        cek_kalimat_input = kalimat_input,
        cek_error         = error_msg,
        total_data        = total,
        positif           = pos,
        negatif           = neg,
        netral            = net,
        akurasi           = akurasi,
        jumlah_train      = jumlah_train,
        jumlah_test       = jumlah_test,
        model_ready       = model_ready,
        cm_data           = cm_data,
        cm_labels         = cm_labels,
        report_data       = report_data,
        presisi_w         = presisi_w,
        recall_w          = recall_w,
        f1_w              = f1_w,
        cv_acc_scores     = _json2.dumps(cv_acc_scores),
        cv_f1_scores      = _json2.dumps(cv_f1_scores),
        cv_acc_mean       = cv_acc_mean,
        cv_acc_std        = cv_acc_std,
        cv_f1_mean        = cv_f1_mean,
    )

# =====================
# CEK KALIMAT (ADMIN)
@app.route('/kalimat', methods=['GET', 'POST'])
@login_required
def kalimat():
    hasil         = None
    kalimat_input = None

    if request.method == 'POST':
        teks = request.form.get('kalimat', '').strip()
        if not teks:
            flash("Masukkan kalimat terlebih dahulu!")
            return render_template('kalimat.html', hasil=None, kalimat_input=None)
        if not os.path.exists(MODEL_PATH):
            flash("Model belum ditraining! Silakan ke halaman Klasifikasi dahulu.")
            return render_template('kalimat.html', hasil=None, kalimat_input=None)

        pipeline = _get_pipeline()
        if pipeline is None:
            flash("Model belum ditraining! Silakan ke halaman Klasifikasi dahulu.")
            return render_template('kalimat.html', hasil=None, kalimat_input=None)

        # === RULE-BASED DULU, baru model ===
        hasil = rule_based_sentimen(teks)
        if hasil is None:
            clean = preprocessor.proses_lengkap(teks)
            hasil = pipeline.predict([clean])[0]

        # Simpan ke session
        session['kalimat_terakhir'] = teks
        session['hasil_terakhir']   = hasil
        kalimat_input = teks

    else:
        hasil         = session.get('hasil_terakhir')
        kalimat_input = session.get('kalimat_terakhir')

    return render_template('kalimat.html', hasil=hasil, kalimat_input=kalimat_input)

# =====================
# LOGOUT
# =====================
@app.route('/logout')
def logout():
    session.clear()
    flash("Anda berhasil logout.")
    return redirect(url_for('login'))

# =====================
# RUN
# =====================
if __name__ == '__main__':
    app.run(debug=True)

# =====================
# API: PREPROCESS
# =====================
@app.route('/api/preprocess', methods=['POST'])
def api_preprocess():
    data = request.get_json(force=True)
    teks = data.get('teks', '').strip()
    if not teks:
        return jsonify({'error': 'Teks kosong'}), 400

    steps = []

    after_emoji = preprocessor.konversi_emoji(teks)
    steps.append({
        'nama': 'Konversi Emoji',
        'detail': 'Emoji dikonversi ke deskripsi teks bahasa Indonesia.',
        'input': teks, 'output': after_emoji, 'is_list': False,
    })

    after_clean = preprocessor.bersihkan_teks(after_emoji)
    steps.append({
        'nama': 'Cleaning',
        'detail': 'Menghapus HTML, karakter khusus, dan angka tidak relevan.',
        'input': after_emoji, 'output': after_clean, 'is_list': False,
    })

    after_case = preprocessor.case_folding(after_clean)
    steps.append({
        'nama': 'Case Folding',
        'detail': 'Seluruh teks diubah menjadi huruf kecil (lowercase).',
        'input': after_clean, 'output': after_case, 'is_list': False,
    })

    after_slang = preprocessor.normalisasi_slang(after_case)
    steps.append({
        'nama': 'Normalisasi Slang',
        'detail': 'Kata slang dan singkatan diubah ke bentuk baku menggunakan kamus slang.',
        'input': after_case, 'output': after_slang, 'is_list': False,
    })

    tokens = preprocessor.tokenisasi(after_slang)
    steps.append({
        'nama': 'Tokenisasi',
        'detail': 'Kalimat dipecah menjadi token (kata) individual.',
        'input': after_slang, 'output': tokens, 'is_list': True,
    })

    after_stop = preprocessor.hapus_stopword(tokens)
    steps.append({
        'nama': 'Hapus Stopword',
        'detail': 'Kata-kata umum yang tidak bermakna (stopword) dihapus.',
        'input': tokens, 'output': after_stop, 'is_list': True,
    })

    after_stem = preprocessor.stemming(after_stop)
    steps.append({
        'nama': 'Stemming',
        'detail': 'Kata dikembalikan ke bentuk dasar menggunakan PySastrawi.',
        'input': after_stop, 'output': after_stem, 'is_list': False,
    })

    return jsonify({'steps': steps})

# =====================
# API: PREDICT
# =====================
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def api_predict():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        resp = app.make_default_options_response()
        resp.headers['Access-Control-Allow-Origin']  = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp

    data = request.get_json(force=True)
    teks = data.get('teks', '').strip()
    if not teks:
        return jsonify({'error': 'Teks kosong'}), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model belum ditraining. Silakan login sebagai admin dan lakukan training terlebih dahulu.'}), 503

    pipeline = _get_pipeline()
    if pipeline is None:
        return jsonify({'error': 'Model belum ditraining. Silakan login sebagai admin dan lakukan training terlebih dahulu.'}), 503

    teks_bersih = preprocessor.proses_lengkap(teks)

    # Coba rule-based dulu (sama seperti admin kalimat.html)
    rb_result = rule_based_sentimen(teks)
    if rb_result is not None:
        label  = rb_result
        source = 'rule-based'
    else:
        label  = pipeline.predict([teks_bersih])[0]
        source = 'model'

    proba_raw = pipeline.predict_proba([teks_bersih])[0]
    classes   = list(pipeline.classes_)
    prob_dict = {c: round(float(p), 4) for c, p in zip(classes, proba_raw)}

    return jsonify({
        'label'       : label,
        'teks_bersih' : teks_bersih,
        'probabilitas': prob_dict,
        'source'      : source,
    })

# =====================
# API: STATS untuk halaman user
# =====================
@app.route('/api/stats')
def api_stats():
    import json as _json

    # Distribusi label dari dataset
    pos = neg = net = total = 0
    preprocessing_path = os.path.join(UPLOAD_FOLDER, 'hasil_preprocessing.csv')
    dataset_path = preprocessing_path if os.path.exists(preprocessing_path) else get_dataset_file()
    if dataset_path and os.path.exists(dataset_path):
        try:
            df = read_csv_smart(dataset_path)
            total = len(df)
            if 'label' in df.columns:
                lbl = df['label'].astype(str).str.strip().str.lower()
                pos = int((lbl == 'positif').sum())
                neg = int((lbl == 'negatif').sum())
                net = int((lbl == 'netral').sum())
        except Exception:
            pass

    # Hasil training dari file JSON
    akurasi = jumlah_train = jumlah_test = 0
    presisi_w = recall_w = f1_w = 0
    report_data = cm_data = cm_labels = None
    cv_acc_scores = cv_f1_scores = []
    cv_acc_mean = cv_acc_std = cv_f1_mean = 0
    model_ready = os.path.exists(MODEL_PATH)

    result_path = os.path.join(MODEL_FOLDER, 'training_result.json')
    if os.path.exists(result_path):
        try:
            with open(result_path, encoding='utf-8') as _f:
                saved = _json.load(_f)
            akurasi       = saved.get('akurasi', 0)
            jumlah_train  = saved.get('jumlah_train', 0)
            jumlah_test   = saved.get('jumlah_test', 0)
            report_data   = saved.get('report_data')
            cm_data       = saved.get('cm_data')
            cm_labels     = saved.get('cm_labels')
            presisi_w     = saved.get('presisi_w', 0)
            recall_w      = saved.get('recall_w', 0)
            f1_w          = saved.get('f1_w', 0)
            cv_acc_scores = saved.get('cv_acc_scores', [])
            cv_f1_scores  = saved.get('cv_f1_scores', [])
            cv_acc_mean   = saved.get('cv_acc_mean', 0)
            cv_acc_std    = saved.get('cv_acc_std', 0)
            cv_f1_mean    = saved.get('cv_f1_mean', 0)
        except Exception:
            pass

    return jsonify({
        'model_ready'   : model_ready,
        'total_data'    : total,
        'positif'       : pos,
        'negatif'       : neg,
        'netral'        : net,
        'akurasi'       : round(float(akurasi), 2),
        'presisi_w'     : round(float(presisi_w), 2),
        'recall_w'      : round(float(recall_w), 2),
        'f1_w'          : round(float(f1_w), 2),
        'jumlah_train'  : jumlah_train,
        'jumlah_test'   : jumlah_test,
        'report_data'   : report_data,
        'cm_data'       : cm_data,
        'cm_labels'     : cm_labels,
        'cv_acc_scores' : cv_acc_scores,
        'cv_f1_scores'  : cv_f1_scores,
        'cv_acc_mean'   : round(float(cv_acc_mean), 2),
        'cv_acc_std'    : round(float(cv_acc_std), 2),
        'cv_f1_mean'    : round(float(cv_f1_mean), 2),
    })