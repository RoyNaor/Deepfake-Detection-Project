import os
import shutil
import random

# --- הגדרות (חובה לשנות את השורה הראשונה!) ---

# 1. איפה נמצאת התיקייה שחילצת מה-ZIP? (נתיב מלא)
# שימי לב: זה הנתיב לתיקייה הראשית שנוצרה אחרי החילוץ
SOURCE_ROOT = r"C:\Users\YourName\Downloads\ASVspoof2019_LA" 

# 2. איפה הפרויקט שלך נמצא?
PROJECT_ROOT = r"./" # זה אומר "בתיקייה הנוכחית"

# 3. כמה קבצים אנחנו רוצים?
NUM_TRAIN_REAL = 2000
NUM_TRAIN_FAKE = 2000
NUM_TEST_REAL = 500
NUM_TEST_FAKE = 500

# --- נתיבים פנימיים (אל תשני אלא אם כן את יודעת מה את עושה) ---
ORIGINAL_PROTO = os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
ORIGINAL_AUDIO = os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_train", "flac")

DEST_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "raw_audio", "train")
DEST_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "raw_audio", "test")
DEST_PROTO_DIR = os.path.join(PROJECT_ROOT, "data", "protocols")

def setup_data():
    print("--- מתחיל בארגון הדאטה ---")
    
    # 1. בדיקה שהמקור קיים
    if not os.path.exists(ORIGINAL_PROTO):
        print(f"שגיאה: לא מצאתי את קובץ הפרוטוקול בנתיב:\n{ORIGINAL_PROTO}")
        return
    if not os.path.exists(ORIGINAL_AUDIO):
        print(f"שגיאה: לא מצאתי את תיקיית האודיו בנתיב:\n{ORIGINAL_AUDIO}")
        return

    # 2. יצירת תיקיות היעד
    os.makedirs(DEST_TRAIN_DIR, exist_ok=True)
    os.makedirs(DEST_TEST_DIR, exist_ok=True)
    os.makedirs(DEST_PROTO_DIR, exist_ok=True)

    # 3. קריאת הפרוטוקול ומיון לקבוצות
    print("קורא את רשימת הקבצים המקורית...")
    real_files = []
    fake_files = []
    
    with open(ORIGINAL_PROTO, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[1]
            label = parts[-1] # 'bonafide' or 'spoof'
            
            if label == 'bonafide':
                real_files.append(filename)
            else:
                fake_files.append(filename)

    # ערבוב אקראי כדי למנוע הטיות
    random.shuffle(real_files)
    random.shuffle(fake_files)
    
    print(f"נמצאו סך הכל: {len(real_files)} אמיתיים, {len(fake_files)} מזויפים.")

    # 4. חלוקה ל-Train ו-Test
    # Train
    train_real = real_files[:NUM_TRAIN_REAL]
    train_fake = fake_files[:NUM_TRAIN_FAKE]
    
    # Test (לוקחים מהסוף כדי שלא יהיו כפילויות עם האימון!)
    test_real = real_files[-NUM_TEST_REAL:]
    test_fake = fake_files[-NUM_TEST_FAKE:]

    print(f"נבחרו לאימון: {len(train_real)} אמיתיים, {len(train_fake)} מזויפים.")
    print(f"נבחרו לטסט: {len(test_real)} אמיתיים, {len(test_fake)} מזויפים.")

    # 5. פונקציית העתקה ויצירת פרוטוקול חדש
    def process_subset(file_list, dest_dir, protocol_name, label_map):
        new_proto_path = os.path.join(DEST_PROTO_DIR, protocol_name)
        
        print(f"מעתיק קבצים ל-{dest_dir}...")
        with open(new_proto_path, 'w') as f_proto:
            for fname in file_list:
                # העתקת הקובץ הפיזי
                src_path = os.path.join(ORIGINAL_AUDIO, fname + ".flac")
                dst_path = os.path.join(dest_dir, fname + ".flac")
                
                try:
                    shutil.copy2(src_path, dst_path)
                    
                    # קביעת התווית (אם הקובץ ברשימת האמיתיים או המזויפים)
                    # נכתוב את זה בפורמט פשוט: filename label (0 or 1)
                    # 1 = Real, 0 = Fake (או להפך, איך שנוח לך, כאן נעשה: bonafide=1, spoof=0)
                    is_real = 1 if fname in label_map['real'] else 0
                    label_str = "bonafide" if is_real else "spoof"
                    
                    f_proto.write(f"{fname} {label_str}\n")
                    
                except FileNotFoundError:
                    print(f"Warning: File {fname} not found via path {src_path}")

    # ביצוע העתקה לאימון
    # מאחדים את הרשימות
    train_all = train_real + train_fake
    random.shuffle(train_all) # מערבבים שוב כדי שהפרוטוקול לא יהיה מסודר מדי
    
    # יוצרים מילון עזר לזיהוי מהיר
    real_set = set(real_files)
    label_map = {'real': real_set} # כל מה שלא פה הוא fake
    
    process_subset(train_all, DEST_TRAIN_DIR, "train_protocol.txt", label_map)
    
    # ביצוע העתקה לטסט
    test_all = test_real + test_fake
    random.shuffle(test_all)
    process_subset(test_all, DEST_TEST_DIR, "test_protocol.txt", label_map)

    print("\n--- סיימנו! ---")
    print(f"הקבצים הועתקו לתיקיית data/raw_audio")
    print(f"נוצרו קבצי פרוטוקול חדשים ב-data/protocols")

if __name__ == "__main__":
    setup_data()