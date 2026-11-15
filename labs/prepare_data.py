"""Build the Sawaqi Water Company case study: synthetic, deterministic, standard library only.

    uv run python prepare_data.py

Writes into data/:
    customers.csv          700 synthetic customers (names, phones and account numbers are invented)
    messages.csv           2,000 customer messages in four varieties, with department, complaint
                           type and the gold account number and district
    split.csv              the split by customer: train (labelled for routing) / val / test, and
                           pool (the rest of the archive, used without its routing labels)
    parallel_50.csv        50 messages written in all four varieties with the same facts
    queries.jsonl          40 archive-search queries (10 per variety) with their relevant messages
    extraction_200.jsonl   200 test messages with gold fields
    smoke_test.jsonl       24 messages for the service's smoke test (4 edge cases)
    reply_tasks.jsonl      10 reply tasks with the facts a reply may use
    rules/R-01 ... R-10    the tariff rule book, English and Arabic
    rule_questions.jsonl   16 questions on the rule book, 4 of which it does not answer

Every run writes the same bytes. The planted problems are listed in solutions/answer_key.md.
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

SEED = 301
DATA = Path(__file__).resolve().parent / "data"
VARIETIES = ["msa", "gulf", "arabizi", "en"]
VARIETY_SHARE = {"msa": 0.35, "gulf": 0.30, "arabizi": 0.15, "en": 0.20}

DEPARTMENTS = ["billing", "payments", "leaks", "supply", "quality", "meters", "connections", "wastewater"]
TYPES = {
    "high_bill": "billing", "duplicate_charge": "billing",
    "payment_failed": "payments", "installment_request": "payments",
    "leak_street": "leaks", "leak_meter": "leaks",
    "no_water": "supply", "low_pressure": "supply",
    "discoloured": "quality", "bad_smell": "quality",
    "meter_fault": "meters", "meter_reading": "meters",
    "new_connection": "connections", "ownership_transfer": "connections",
    "sewage_overflow": "wastewater", "sewage_blockage": "wastewater",
}
# How often each department writes in; Arabizi writers rarely ask about connections or wastewater
DEPT_WEIGHT = {"billing": 18, "payments": 11, "leaks": 13, "supply": 17, "quality": 10, "meters": 13, "connections": 8, "wastewater": 10}
ARABIZI_RARE = {"connections": 0.15, "wastewater": 0.2}

# key: (Arabic, Arabizi, English)
DISTRICTS = {
    "ghadeer": ("الغدير", "el ghadeer", "Al-Ghadeer"),
    "manhal": ("المنهل", "el manhal", "Al-Manhal"),
    "shurfa": ("الشرفة", "el shurfa", "Al-Shurfa"),
    "wurood": ("الورود", "el wurood", "Al-Wurood"),
    "yasmeen": ("الياسمين", "el yasmeen", "Al-Yasmeen"),
    "qalaa": ("القلعة", "el qal3a", "Al-Qalaa"),
    "rabwa": ("الربوة", "el rabwa", "Al-Rabwa"),
    "naseem": ("النسيم", "el naseem", "Al-Naseem"),
}

# 50 parallel templates: (complaint type, MSA, Gulf, Arabizi, English)
TEMPLATES = [
    ("high_bill", "فاتورة هذا الشهر مرتفعة جدا مقارنة بالأشهر السابقة في حي {dist}", "الفاتورة هالشهر مره غاليه مو طبيعي، احنا في {dist}", "el fatoora hal shahar mara ghalya mo 6abee3i, e7na fi {dist}", "This month's bill is much higher than usual, we live in {dist}"),
    ("high_bill", "وصلتني فاتورة بمبلغ {amt} ريال ولا أعرف سبب هذا الارتفاع", "جتني فاتوره {amt} ريال وش السالفه؟", "jatni fatoora {amt} riyal wesh el salfa?", "I got a bill for {amt} SAR and I don't understand why it is so high"),
    ("high_bill", "أعترض على قيمة الفاتورة الأخيرة لأن استهلاكنا لم يتغير", "معترض على الفاتوره الاخيره، استهلاكنا نفسه ما تغير", "m3tared 3la el fatoora el a5eera, esthlakna nafsah ma taghayar", "I dispute my last bill, our usage has not changed"),
    ("duplicate_charge", "تم خصم مبلغ الفاتورة مرتين من حسابي البنكي", "انخصم مني مبلغ الفاتوره مرتين", "ankhasam mni mablagh el fatoora martain", "I was charged twice for the same bill"),
    ("duplicate_charge", "ظهرت الفاتورة نفسها مكررة في كشف الحساب", "الفاتوره طالعه لي مكرره بالكشف", "el fatoora 6al3a li mkarara bil kashf", "The same bill appears twice on my statement"),
    ("duplicate_charge", "دفعت {amt} ريال مرتين وأطلب استرداد المبلغ المكرر", "دفعت {amt} ريال مرتين ابي ترجعون الزياده", "dfa3t {amt} riyal martain abi tarj3oon el ziyada", "I paid {amt} SAR twice and want the duplicate refunded"),
    ("payment_failed", "حاولت السداد عبر التطبيق لكن العملية فشلت", "حاولت اسدد من التطبيق وطلع لي خطأ", "7awalt asadid min el ta6beeq w 6ala3 li 5a6a", "I tried to pay in the app but the payment failed"),
    ("payment_failed", "خصم المبلغ من البطاقة لكن الفاتورة ما زالت غير مدفوعة", "انسحب المبلغ من البطاقه بس الفاتوره للحين مو مدفوعه", "ansa7ab el mablagh min el ba6aqa bs el fatoora lil7een mo madfoo3a", "The money left my card but the bill still shows unpaid"),
    ("payment_failed", "لا تظهر خيارات الدفع في الموقع منذ يومين", "خيارات الدفع ما تطلع بالموقع من يومين", "5iyarat el daf3 ma t6la3 bil mawqi3 min yomain", "The payment options have not shown on the website for two days"),
    ("installment_request", "أرغب في تقسيط مبلغ {amt} ريال على عدة أشهر", "ابي اقسط مبلغ {amt} ريال على كم شهر", "abi aqasit mablagh {amt} riyal 3la kam shahar", "I would like to pay {amt} SAR in monthly instalments"),
    ("installment_request", "هل يمكن جدولة المتأخرات لأن دخلي محدود", "ممكن تجدولون المتأخرات؟ الدخل محدود", "mumkin tjadwloon el muta2a5irat? el da5l ma7dood", "Can you schedule my arrears? My income is limited"),
    ("installment_request", "أطلب خطة سداد للفاتورة المتأخرة", "ابغى خطة سداد للفاتوره المتأخره", "abgha 5i6at sadad lil fatoora el mit2a5ra", "I am asking for a payment plan for my overdue bill"),
    ("leak_street", "يوجد تسرب مياه كبير في الشارع أمام منزلي في حي {dist}", "فيه تسريب ماي قوي بالشارع قدام بيتي في {dist}", "fee tasreeb mai gawi bil share3 gidam baiti fi {dist}", "There is a big water leak in the street outside my house in {dist}"),
    ("leak_street", "الماء يتدفق من أنبوب مكسور عند تقاطع {dist} منذ {days} أيام", "الماي يطلع من ماسوره مكسوره عند تقاطع {dist} من {days} ايام", "el mai y6la3 min masoora maksoora 3ind ta2a6o3 {dist} min {days} ayam", "Water has been pouring from a broken pipe at the {dist} junction for {days} days"),
    ("leak_street", "هناك هدر للمياه في الحديقة العامة بسبب كسر في الخط", "فيه ماي ضايع بالحديقه العامه بسبب كسر بالخط", "fee mai 9aye3 bil 7adeeqa bsabab kasr bil 5a6", "Water is being wasted in the public park because of a broken main"),
    ("leak_meter", "يوجد تسرب عند العداد خارج المنزل", "فيه تسريب عند العداد برا البيت", "fee tasreeb 3ind el 3adad barra el bait", "There is a leak at the meter outside the house"),
    ("leak_meter", "الماء يتسرب من وصلة العداد ويبلل الرصيف", "الماي يسرب من وصلة العداد والرصيف كله مويه", "el mai ysarib min wislat el 3adad wel ra9eef kilah moya", "Water is dripping from the meter connection and the pavement is wet"),
    ("leak_meter", "صندوق العداد ممتلئ بالماء منذ أسبوع", "صندوق العداد مليان ماي من اسبوع", "9andooq el 3adad malyan mai min usboo3", "The meter box has been full of water for a week"),
    ("no_water", "انقطعت المياه عن منزلنا في حي {dist} منذ الصباح", "الماي مقطوع عندنا في {dist} من الصبح", "el mai maqtoo3 3indna fi {dist} min el sob7", "We have had no water at home in {dist} since the morning"),
    ("no_water", "لا توجد مياه في الحي بأكمله منذ {days} أيام", "الحي كله ما فيه ماي من {days} ايام", "el 7ai kilah ma feeh mai min {days} ayam", "The whole neighbourhood has had no water for {days} days"),
    ("no_water", "متى ستعود المياه؟ لم يصلنا أي إشعار بالانقطاع", "متى يرجع الماي؟ محد بلغنا عن القطع", "mata yirja3 el mai? ma7ad ballaghna 3an el qa6i3", "When will the water come back? Nobody told us about the cut"),
    ("no_water", "خزان المنزل فارغ والمياه مقطوعة ولدينا كبار سن", "الخزان فاضي والماي مقطوع وعندنا كبار سن", "el 5azan fa9i wel mai maqtoo3 w 3indna kibar sin", "Our tank is empty, the water is off and we have elderly people at home"),
    ("low_pressure", "ضغط المياه ضعيف جدا في الطابق الثاني", "ضغط الماي مره ضعيف بالدور الثاني", "6ag6 el mai mara 9a3eef bil door el thani", "The water pressure is very weak on the second floor"),
    ("low_pressure", "المياه تصل ضعيفة في أوقات الذروة في {dist}", "الماي يجي ضعيف وقت الذروه في {dist}", "el mai yiji 9a3eef wagt el thurwa fi {dist}", "The water comes through weakly at peak times in {dist}"),
    ("low_pressure", "الخزان لا يمتلئ بسبب ضعف الضغط", "الخزان ما يتعبى من ضعف الضغط", "el 5azan ma yit3abba min 9a3f el 6ag6", "Our tank does not fill because the pressure is low"),
    ("discoloured", "المياه تخرج بلون بني من الصنبور", "الماي يطلع لونه بني من الحنفيه", "el mai y6la3 loonah bunni min el 7anafiya", "The water coming out of the tap is brown"),
    ("discoloured", "لون الماء أصفر بعد عودة الإمداد في حي {dist}", "الماي صاير لونه اصفر بعد ما رجع في {dist}", "el mai 9ayer loonah a9far ba3d ma rija3 fi {dist}", "The water turned yellow after the supply came back in {dist}"),
    ("discoloured", "توجد رواسب في المياه ونخشى على الأطفال", "فيه رواسب بالماي وخايفين على العيال", "fee rawasib bil mai w 5aifeen 3la el 3iyal", "There is sediment in the water and we worry about the children"),
    ("bad_smell", "للمياه رائحة كريهة تشبه رائحة الكلور", "الماي ريحته خايسه كأنها كلور", "el mai re7tah 5aisa ka2anha chlor", "The water smells bad, like chlorine"),
    ("bad_smell", "طعم الماء غريب منذ {days} أيام", "طعم الماي غريب من {days} ايام", "6a3m el mai ghareeb min {days} ayam", "The water has tasted strange for {days} days"),
    ("bad_smell", "رائحة المياه في {dist} لا تحتمل", "ريحة الماي في {dist} ما تنطاق", "re7at el mai fi {dist} ma tin6aq", "The smell of the water in {dist} is unbearable"),
    ("meter_fault", "العداد لا يتحرك رغم استخدام المياه", "العداد واقف مع اننا نستخدم الماي", "el 3adad waqif ma3 ina nista5dim el mai", "The meter does not move even when we use water"),
    ("meter_fault", "شاشة العداد الذكي مطفأة", "شاشة العداد الذكي طافيه", "shashat el 3adad el thaki 6afya", "The smart meter's screen is blank"),
    ("meter_fault", "أطلب فحص العداد لأنه يسجل استهلاكا غير صحيح", "ابي تفحصون العداد يحسب غلط", "abi tif7soon el 3adad y7sib ghala6", "Please test my meter, it records the wrong usage"),
    ("meter_fault", "الفاتورة مرتفعة وأعتقد أن العداد معطل", "الفاتوره غاليه واحس العداد خربان", "el fatoora ghalya w a7is el 3adad 5arban", "My bill is high and I think the meter is broken"),
    ("meter_reading", "القراءة في الفاتورة لا تطابق قراءة العداد", "القراءه بالفاتوره غير عن قراءة العداد", "el qira2a bil fatoora ghair 3an qira2at el 3adad", "The reading on the bill does not match the meter"),
    ("meter_reading", "أرسل لكم قراءة العداد الحالية وهي {reading}", "هذي قراءة العداد الحين {reading}", "hathi qira2at el 3adad el 7een {reading}", "Here is my current meter reading: {reading}"),
    ("meter_reading", "لم يأت قارئ العداد منذ شهرين", "قارئ العداد ما جا من شهرين", "qari2 el 3adad ma ja min shahrain", "The meter reader has not come for two months"),
    ("new_connection", "أرغب في توصيل المياه لمنزل جديد في حي {dist}", "ابي توصيل ماي لبيت جديد في {dist}", "abi taw9eel mai l bait jdeed fi {dist}", "I want a water connection for a new house in {dist}"),
    ("new_connection", "ما هي متطلبات طلب عداد جديد لمحل تجاري", "وش المطلوب عشان عداد جديد لمحل؟", "wesh el ma6loob 3ashan 3adad jdeed l ma7al?", "What do I need to apply for a new meter for a shop?"),
    ("new_connection", "قدمت طلب توصيل منذ {days} أيام ولم يتم الرد", "قدمت طلب توصيل من {days} ايام ومحد رد", "gaddamt 6alab taw9eel min {days} ayam w ma7ad rad", "I applied for a connection {days} days ago and have had no reply"),
    ("ownership_transfer", "اشتريت المنزل وأرغب في نقل الحساب باسمي", "شريت البيت وابي انقل الحساب باسمي", "shareet el bait w abi anqil el 7sab bismi", "I bought the house and want the account moved to my name"),
    ("ownership_transfer", "أطلب إغلاق الحساب لأنني انتقلت من الشقة", "ابي اسكر الحساب طلعت من الشقه", "abi askir el 7sab 6la3t min el shiqa", "Please close my account, I have moved out of the flat"),
    ("ownership_transfer", "كيف أنقل عداد المياه إلى المستأجر الجديد", "كيف انقل العداد للمستأجر الجديد؟", "kaif anqil el 3adad lil musta2jir el jdeed?", "How do I transfer the meter to the new tenant?"),
    ("sewage_overflow", "مياه الصرف الصحي تطفح في الشارع في حي {dist}", "المجاري طافحه بالشارع في {dist}", "el majari 6af7a bil share3 fi {dist}", "Sewage is overflowing into the street in {dist}"),
    ("sewage_overflow", "غرفة التفتيش تفيض أمام المدرسة", "المنهل فايض قدام المدرسه", "el manhal fayi9 gidam el madrasa", "The manhole in front of the school is overflowing"),
    ("sewage_overflow", "رائحة الصرف الصحي قوية وتوجد مياه راكدة", "ريحة المجاري قويه وفيه مويه واقفه", "re7at el majari gawya w fee moya waqfa", "There is a strong sewage smell and standing water"),
    ("sewage_blockage", "يوجد انسداد في خط الصرف الخاص بالمنزل", "البيت فيه انسداد بخط المجاري", "el bait feeh insidad b 5a6 el majari", "The sewer line of the house is blocked"),
    ("sewage_blockage", "المياه لا تنصرف من المطبخ والحمام", "المويه ما تنزل بالمطبخ والحمام", "el moya ma tinzil bil ma6ba5 wel 7ammam", "Water is not draining from the kitchen and bathroom"),
    ("sewage_blockage", "أطلب سيارة شفط لأن البيارة ممتلئة", "ابي وايت شفط البياره فل", "abi wait shaf6 el bayara full", "I need a suction truck, the septic tank is full"),
]
assert len(TEMPLATES) == 50

# Short archive-search phrases per complaint type: (MSA, Gulf, Arabizi, English)
QUERY_PHRASES = {
    "high_bill": ("فواتير مرتفعة", "فاتوره غاليه", "fatoora ghalya", "high water bill"),
    "duplicate_charge": ("خصم مكرر للفاتورة", "انخصم مرتين", "ankhasam martain", "charged twice"),
    "payment_failed": ("فشل عملية السداد", "الدفع ما زبط", "el daf3 ma zaba6", "payment failed"),
    "installment_request": ("طلب تقسيط", "ابي اقسط", "abi aqasit", "instalment plan"),
    "leak_street": ("تسرب في الشارع", "تسريب ماي بالشارع", "tasreeb mai bil share3", "street leak"),
    "leak_meter": ("تسرب عند العداد", "تسريب عند العداد", "tasreeb 3ind el 3adad", "leak at the meter"),
    "no_water": ("انقطاع المياه", "الماي مقطوع", "el mai maqtoo3", "no water"),
    "low_pressure": ("ضعف ضغط المياه", "ضغط الماي ضعيف", "6ag6 el mai 9a3eef", "low water pressure"),
    "discoloured": ("تغير لون المياه", "لون الماي متغير", "loon el mai mitghayir", "brown or yellow water"),
    "bad_smell": ("رائحة المياه", "ريحة الماي", "re7at el mai", "water smells bad"),
    "meter_fault": ("عطل العداد", "العداد خربان", "el 3adad 5arban", "broken meter"),
    "meter_reading": ("قراءة العداد", "قراءة العداد غلط", "qira2at el 3adad", "meter reading"),
    "new_connection": ("توصيل جديد", "توصيل ماي جديد", "taw9eel jdeed", "new connection"),
    "ownership_transfer": ("نقل ملكية الحساب", "انقل الحساب باسمي", "anqil el 7sab", "transfer the account"),
    "sewage_overflow": ("طفح الصرف الصحي", "المجاري طافحه", "el majari 6af7a", "sewage overflow"),
    "sewage_blockage": ("انسداد الصرف", "انسداد المجاري", "insidad el majari", "blocked drain"),
}

GREETINGS = {
    "msa": ["السلام عليكم،", "مرحبا،", "تحية طيبة،"],
    "gulf": ["السلام عليكم", "هلا", "مساء الخير يا جماعه", "لو سمحتو"],
    "arabizi": ["salam", "hala", "slm 3likom", "law sama7tu"],
    "en": ["Hello,", "Hi,", "Good evening,", "Dear Sawaqi,"],
}
CLOSINGS = {
    "msa": ["أرجو المساعدة.", "شكرا لكم.", "أرجو الرد بسرعة."],
    "gulf": ["تكفون ساعدونا", "الله يعطيكم العافيه", "ردوا علي بسرعه"],
    "arabizi": ["tkfoon sa3dona", "allah y3afeekom", "rdo 3lay bsr3a"],
    "en": ["Please help.", "Thanks.", "Please reply soon."],
}
ACCOUNT_PHRASES = {
    "msa": ["رقم الحساب {acc}", "حسابي رقم {acc}"],
    "gulf": ["رقم حسابي {acc}", "الحساب {acc}"],
    "arabizi": ["rqm el 7sab {acc}", "7sabi {acc}"],
    "en": ["account number {acc}", "acct {acc}", "my account is {acc}"],
}
PHONE_PHRASES = {
    "msa": ["للتواصل {phone}", "رقم جوالي {phone}"],
    "gulf": ["جوالي {phone}", "كلموني على {phone}"],
    "arabizi": ["jawali {phone}", "kalmoni 3la {phone}"],
    "en": ["call me on {phone}", "my mobile is {phone}"],
}
NAME_PHRASES = {"msa": "أنا {name}،", "gulf": "معك {name}", "arabizi": "m3ak {name}", "en": "This is {name}."}
DIST_SUFFIX = {"msa": "نحن في حي {dist}", "gulf": "احنا في {dist}", "arabizi": "e7na fi {dist}", "en": "We are in {dist}"}

FIRST_AR = ["محمد", "عبدالله", "فهد", "خالد", "سلطان", "ناصر", "يوسف", "مشاري", "بدر", "حمد", "منيرة", "العنود", "مها", "ريم", "هيفاء", "جواهر", "شهد", "دانة", "لولوة", "أسماء"]
FIRST_LAT = ["Mohammed", "Abdullah", "Fahad", "Khalid", "Sultan", "Nasser", "Yousef", "Meshari", "Badr", "Hamad", "Munira", "Alanoud", "Maha", "Reem", "Haifa", "Jawaher", "Shahad", "Dana", "Lulwa", "Asma"]
LAST_AR = ["الدوسري", "القحطاني", "العتيبي", "الشهري", "الزهراني", "الحربي", "العنزي", "المالكي", "السبيعي", "الرشيدي", "البقمي", "الشمري"]
LAST_LAT = ["Al-Dosari", "Al-Qahtani", "Al-Otaibi", "Al-Shehri", "Al-Zahrani", "Al-Harbi", "Al-Anazi", "Al-Malki", "Al-Subaie", "Al-Rashidi", "Al-Buqami", "Al-Shammari"]

ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
DIACRITICS = ["َ", "ُ", "ِ", "ْ", "ّ"]


def dist_text(key: str, variety: str) -> str:
    ar, lat, en = DISTRICTS[key]
    return {"msa": ar, "gulf": ar, "arabizi": lat, "en": en}[variety]


def add_noise(text: str, variety: str, rng: random.Random) -> str:
    """Spelling variation real messages carry: diacritics, tatweel, alef and ta marbuta forms."""
    if variety in ("msa", "gulf"):
        words = text.split(" ")
        out = []
        for w in words:
            if any(ch.isdigit() for ch in w):
                out.append(w)
                continue
            if variety == "msa" and rng.random() < 0.06 and len(w) > 2:
                i = rng.randrange(1, len(w))
                w = w[:i] + rng.choice(DIACRITICS) + w[i:]
            if rng.random() < 0.04 and len(w) > 3:
                i = rng.randrange(1, len(w) - 1)
                w = w[:i] + "ـ" + w[i:]
            if w.startswith(("أ", "إ")) and rng.random() < 0.35:
                w = "ا" + w[1:]
            elif w.startswith("ا") and rng.random() < 0.08:
                w = "أ" + w[1:]
            if w.endswith("ة") and rng.random() < (0.15 if variety == "msa" else 0.5):
                w = w[:-1] + "ه"
            if w.endswith("ي") and rng.random() < 0.1:
                w = w[:-1] + "ى"
            out.append(w)
        text = " ".join(out)
    elif variety == "arabizi":
        if rng.random() < 0.5:
            text = text.replace("el ", "al ")
        if rng.random() < 0.3:
            text = text.replace("ee", "i")
        if rng.random() < 0.2:
            text = text.upper() if rng.random() < 0.2 else text.capitalize()
    else:
        if rng.random() < 0.15:
            text = text.lower()
    return text

# A second clause that is true but is not the complaint: it carries another department's words.
# (department whose words it borrows, MSA, Gulf, Arabizi, English)
ASIDES = [
    ("payments", "علما بأنني سددت الفاتورة السابقة", "وانا مسدد الفاتوره اللي قبل", "w ana msadid el fatoora elli gabil", "and I have already paid the last bill"),
    ("meters", "علما بأن العداد تم تغييره الشهر الماضي", "والعداد توه متغير الشهر اللي فات", "wel 3adad tawah mitghayir el shahar elli fat", "and the meter was replaced last month"),
    ("leaks", "وقد حدث تسرب في الشارع العام الماضي", "وكان فيه تسريب بالشارع السنه اللي فاتت", "w kan fee tasreeb bil share3 el sana elli fatat", "and there was a leak in the street last year"),
    ("supply", "وكانت المياه مقطوعة الأسبوع الماضي أيضا", "والماي كان مقطوع الاسبوع اللي فات بعد", "wel mai kan maqtoo3 el usboo3 elli fat ba3ad", "and the water was also off last week"),
    ("wastewater", "وهناك أعمال صرف صحي في شارعنا", "وفيه شغل مجاري بشارعنا", "w fee shughul majari b share3na", "and there are sewer works in our street"),
    ("billing", "ولم تصلني فاتورة هذا الشهر بعد", "وفاتورة هالشهر للحين ما جتني", "w fatoorat hal shahar lil7een ma jatni", "and this month's bill has not reached me yet"),
]
ARABIZI_SWAPS = [("3", "a"), ("7", "h"), ("5", "kh"), ("6", "t"), ("9", "s"), ("2", ""), ("ee", "i"), ("oo", "u"), ("ai", "ay")]


def vary(template: str, variety: str, rng: random.Random) -> str:
    """Spelling variation on a template before its slots are filled: Arabizi conventions and typos."""
    if variety == "arabizi":
        for a, b in ARABIZI_SWAPS:
            if rng.random() < 0.3:
                template = re.sub(a + r"(?![^{]*})", b, template)
    out = []
    for w in template.split(" "):
        if "{" not in w and len(w) > 3 and not any(ch.isdigit() for ch in w) and rng.random() < 0.1:
            i = rng.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1:] if rng.random() < 0.5 else w[:i - 1] + w[i] + w[i - 1] + w[i + 1:]
        out.append(w)
    return " ".join(out)


def fill(template: str, variety: str, slots: dict) -> str:
    return template.format(dist=dist_text(slots["dist"], variety), amt=slots["amt"], days=slots["days"], reading=slots["reading"])


def make_customers(rng: random.Random) -> list[dict]:
    customers, used_acc, used_phone = [], set(), set()
    varieties = list(VARIETY_SHARE)
    weights = [VARIETY_SHARE[v] for v in varieties]
    for i in range(700):
        while True:
            acc = "30" + "".join(rng.choice("0123456789") for _ in range(8))
            if acc not in used_acc:
                used_acc.add(acc)
                break
        while True:
            phone = "05" + "".join(rng.choice("0123456789") for _ in range(8))
            if phone not in used_phone:
                used_phone.add(phone)
                break
        f, l = rng.randrange(len(FIRST_AR)), rng.randrange(len(LAST_AR))
        customers.append({
            "customer_id": f"C{i + 1:04d}", "account_number": acc, "phone": phone,
            "name_ar": f"{FIRST_AR[f]} {LAST_AR[l]}", "name_lat": f"{FIRST_LAT[f]} {LAST_LAT[l]}",
            "district": rng.choice(list(DISTRICTS)), "variety": rng.choices(varieties, weights)[0],
        })
    return customers


def pick_template(variety: str, rng: random.Random) -> int:
    weights = []
    for t in TEMPLATES:
        dept = TYPES[t[0]]
        n_in_type = sum(1 for x in TEMPLATES if x[0] == t[0])
        w = DEPT_WEIGHT[dept] / 2 / n_in_type
        if variety == "arabizi":
            w *= ARABIZI_RARE.get(dept, 1.0)
        weights.append(w)
    return rng.choices(range(len(TEMPLATES)), weights)[0]


def compose(cust: dict, variety: str, t_idx: int, slots: dict, opts: dict, rng: random.Random) -> tuple[str, dict]:
    ctype, *texts = TEMPLATES[t_idx]
    vi = VARIETIES.index(variety)
    body = fill(vary(texts[vi], variety, rng), variety, slots)
    if opts["aside"] is not None:
        others = [a for a in ASIDES if a[0] != TYPES[ctype]]
        body += " " + vary(others[opts["aside"] % len(others)][1 + vi], variety, rng)
    district = slots["dist"] if "{dist}" in texts[0] else None
    parts = []
    if opts["greet"]:
        parts.append(rng.choice(GREETINGS[variety]))
    if opts["name"]:
        parts.append(NAME_PHRASES[variety].format(name=cust["name_ar"] if variety in ("msa", "gulf") else cust["name_lat"]))
    parts.append(body)
    if opts["dist_suffix"] and district is None:
        parts.append(DIST_SUFFIX[variety].format(dist=dist_text(slots["dist"], variety)))
        district = slots["dist"]
    account = None
    if opts["account"]:
        acc = cust["account_number"]
        shown = acc.translate(ARABIC_INDIC) if (variety in ("msa", "gulf") and opts["indic"]) else acc
        parts.append(rng.choice(ACCOUNT_PHRASES[variety]).format(acc=shown))
        account = acc
    if opts["phone"]:
        parts.append(rng.choice(PHONE_PHRASES[variety]).format(phone=cust["phone"]))
    if opts["close"]:
        parts.append(rng.choice(CLOSINGS[variety]))
    text = add_noise(" ".join(parts), variety, rng)
    gold = {"complaint_type": ctype, "department": TYPES[ctype], "account_number": account, "district": district,
            "has_phone": bool(opts["phone"]), "has_name": bool(opts["name"])}
    return text, gold


def random_slots(cust: dict, rng: random.Random) -> dict:
    return {"dist": cust["district"] if rng.random() < 0.8 else rng.choice(list(DISTRICTS)),
            "amt": rng.choice([380, 425, 610, 745, 890, 1120, 1340, 1675, 2210]),
            "days": rng.choice([2, 3, 4, 5, 6]), "reading": str(rng.randrange(10000, 99999))}


def random_opts(rng: random.Random) -> dict:
    return {"greet": rng.random() < 0.5, "name": rng.random() < 0.2, "dist_suffix": rng.random() < 0.3,
            "account": rng.random() < 0.85, "indic": rng.random() < 0.3, "phone": rng.random() < 0.4, "close": rng.random() < 0.4,
            "aside": rng.randrange(60) if rng.random() < 0.3 else None}


def make_messages(customers: list[dict], rng: random.Random) -> list[dict]:
    """2,000 messages. About one customer in five writes again about the same problem (near-duplicates)."""
    rows = []
    while len(rows) < 2000:
        cust = rng.choice(customers)
        variety = cust["variety"] if rng.random() < 0.9 else rng.choices(VARIETIES, [VARIETY_SHARE[v] for v in VARIETIES])[0]
        t_idx = pick_template(variety, rng)
        slots, opts = random_slots(cust, rng), random_opts(rng)
        repeats = 1 if rng.random() > 0.2 else rng.choice([2, 3, 4])
        for r in range(repeats):
            if len(rows) >= 2000:
                break
            if r:  # the same complaint again: same facts, a different greeting or closing
                opts = dict(opts, greet=rng.random() < 0.5, close=rng.random() < 0.6)
            text, gold = compose(cust, variety, t_idx, slots, opts, rng)
            rows.append({"msg_id": f"M{len(rows) + 1:04d}", "customer_id": cust["customer_id"], "variety": variety,
                         "template": t_idx, "text": text, **gold})
    return rows


def split_by_customer(customers: list[dict], messages: list[dict], rng: random.Random) -> dict:
    ids = sorted({m["customer_id"] for m in messages})
    rng.shuffle(ids)
    counts = {}
    for m in messages:
        counts[m["customer_id"]] = counts.get(m["customer_id"], 0) + 1
    split, n_test, n_val, n_train = {}, 0, 0, 0
    for cid in ids:
        if n_test < 400:
            split[cid] = "test"
            n_test += counts[cid]
        elif n_val < 300:
            split[cid] = "val"
            n_val += counts[cid]
        elif n_train < 400:  # the messages the contact centre labelled by hand
            split[cid] = "train"
            n_train += counts[cid]
        else:  # the rest of the archive: no routing label is used from it
            split[cid] = "pool"
    return split


RULES = [
    ("R-01", "Scope and definitions", "النطاق والتعريفات",
     "This rule book sets the charges and service rules of Sawaqi Water Company for residential and commercial customers. A billing month is one calendar month. Consumption is measured in cubic metres (m³) by the customer's meter.",
     "يحدد هذا الدليل رسوم شركة سواقي للمياه وقواعد خدمتها للعملاء السكنيين والتجاريين. شهر الفوترة شهر ميلادي واحد. يقاس الاستهلاك بالمتر المكعب بعداد العميل."),
    ("R-02", "Residential tariff", "التعرفة السكنية",
     "Residential water is charged per m³ in four tiers each month: 0 to 15 m³ at SAR 1.20, 16 to 30 m³ at SAR 2.50, 31 to 45 m³ at SAR 4.80, and above 45 m³ at SAR 6.40. A meter service charge of SAR 9 is added every month. Wastewater service adds 50% of the water charge where the property is connected to the network.",
     "تحتسب مياه المساكن بالمتر المكعب على أربع شرائح شهريا: من 0 إلى 15 م³ بسعر 1.20 ريال، ومن 16 إلى 30 م³ بسعر 2.50 ريال، ومن 31 إلى 45 م³ بسعر 4.80 ريال، وما زاد على 45 م³ بسعر 6.40 ريال. تضاف رسوم خدمة عداد قدرها 9 ريالات شهريا. وتضاف خدمة الصرف الصحي بنسبة 50% من قيمة المياه للعقار المتصل بالشبكة."),
    ("R-03", "Commercial tariff", "التعرفة التجارية",
     "Commercial customers pay a flat SAR 7.00 per m³ and a meter service charge of SAR 25 per month. Wastewater service adds 60% of the water charge.",
     "يدفع العملاء التجاريون سعرا موحدا قدره 7.00 ريالات للمتر المكعب ورسوم خدمة عداد قدرها 25 ريالا شهريا. وتضاف خدمة الصرف الصحي بنسبة 60% من قيمة المياه."),
    ("R-04", "Payment and late payment", "السداد والتأخر",
     "A bill is due 30 days after it is issued. A bill unpaid after 60 days receives a written notice. Supply may be suspended after 90 days, except for households with a registered medical need. Payment is accepted in the Sawaqi app, on the website and through bank channels.",
     "تستحق الفاتورة بعد 30 يوما من إصدارها. ترسل إشعار كتابي إذا لم تسدد الفاتورة بعد 60 يوما. ويجوز إيقاف الخدمة بعد 90 يوما، إلا للأسر التي لديها حالة طبية مسجلة. يقبل السداد في تطبيق سواقي وعلى الموقع وعبر القنوات البنكية."),
    ("R-05", "Instalment plans", "خطط التقسيط",
     "Arrears above SAR 500 can be paid in up to six monthly instalments. The first instalment is 20% of the arrears and is paid when the plan is signed. A plan lapses if two instalments are missed.",
     "يمكن تقسيط المتأخرات التي تزيد على 500 ريال على ستة أقساط شهرية بحد أقصى. القسط الأول 20% من المتأخرات ويدفع عند توقيع الخطة. وتلغى الخطة إذا فات قسطان."),
    ("R-06", "Leak allowance", "تسوية التسربات",
     "When a customer repairs a leak on their side of the meter within 14 days of finding it, the excess consumption of that bill is recalculated at the first-tier price. The allowance is granted once in any 12 months and needs a plumber's invoice.",
     "إذا أصلح العميل تسربا في الجهة الخاصة به من العداد خلال 14 يوما من اكتشافه، يعاد احتساب الاستهلاك الزائد في تلك الفاتورة بسعر الشريحة الأولى. تمنح التسوية مرة واحدة كل 12 شهرا وتتطلب فاتورة سباك."),
    ("R-07", "Meter testing and disputes", "فحص العداد والاعتراض",
     "A customer may ask for a meter test. The fee is SAR 150 and is refunded if the meter's error is above 3%. If the meter is found faulty, bills are corrected for up to the last three months.",
     "يحق للعميل طلب فحص العداد. رسوم الفحص 150 ريالا وتسترد إذا تجاوز خطأ العداد 3%. وإذا ثبت عطل العداد تصحح الفواتير لمدة أقصاها الأشهر الثلاثة الأخيرة."),
    ("R-08", "Connections, transfers and closing", "التوصيل ونقل الحساب والإغلاق",
     "A new residential connection costs SAR 1,800 and is completed within 15 working days of a complete application. Transferring an account to a new owner is free with the property deed. Closing an account takes a final reading within 3 working days.",
     "تكلفة التوصيل السكني الجديد 1,800 ريال وينفذ خلال 15 يوم عمل من اكتمال الطلب. نقل الحساب إلى مالك جديد مجاني مع صك الملكية. ويتطلب إغلاق الحساب قراءة نهائية خلال 3 أيام عمل."),
    ("R-09", "Interruptions", "انقطاع الخدمة",
     "Planned interruptions are announced 48 hours in advance. When an unplanned interruption lasts more than 24 hours, a household may ask for a free water tanker. For each full day beyond 48 hours, the account is credited SAR 50.",
     "يعلن عن الانقطاعات المجدولة قبل 48 ساعة. وإذا استمر انقطاع غير مجدول أكثر من 24 ساعة يحق للأسرة طلب صهريج مياه مجاني. ويضاف إلى الحساب رصيد 50 ريالا عن كل يوم كامل بعد 48 ساعة."),
    ("R-10", "Customer data", "بيانات العملاء",
     "Customer names, phone numbers and account numbers are personal data. Sawaqi classifies customer records as Restricted under its data classification, following the national Data Classification Policy. Personal fields are masked before message text is sent to any external service, and only the fields a task needs are kept.",
     "أسماء العملاء وأرقام جوالاتهم وأرقام حساباتهم بيانات شخصية. تصنف سواقي سجلات العملاء بمستوى مقيد وفق تصنيف بياناتها، اتباعا لسياسة تصنيف البيانات الوطنية. وتحجب الحقول الشخصية قبل إرسال نص الرسالة إلى أي خدمة خارجية، ولا يحتفظ إلا بالحقول التي تحتاجها المهمة."),
]

RULE_QUESTIONS = [
    ("q01", "en", "How much is one cubic metre in the second residential tier?", "R-02", "2.50"),
    ("q02", "msa", "كم رسوم خدمة العداد الشهرية للمنازل؟", "R-02", "9"),
    ("q03", "en", "What does a commercial customer pay per cubic metre?", "R-03", "7.00"),
    ("q04", "msa", "متى تستحق الفاتورة بعد إصدارها؟", "R-04", "30"),
    ("q05", "gulf", "ابي اقسط المتأخرات، كم القسط الاول؟", "R-05", "20%"),
    ("q06", "en", "How many instalments can arrears be split into?", "R-05", "six"),
    ("q07", "msa", "أصلحت تسربا داخل المنزل، كيف تحسب الفاتورة؟", "R-06", "الشريحة الأولى"),
    ("q08", "en", "How much does a meter test cost and when is it refunded?", "R-07", "150"),
    ("q09", "gulf", "كم يكلف توصيل ماي لبيت جديد؟", "R-08", "1,800"),
    ("q10", "en", "Is transferring an account to a new owner free?", "R-08", "free"),
    ("q11", "msa", "متى يحق لي طلب صهريج مجاني عند الانقطاع؟", "R-09", "24"),
    ("q12", "en", "Can you send my message text to an outside service with my phone number in it?", "R-10", "masked"),
    ("q13", "en", "Is there a discount on water for farms?", None, None),
    ("q14", "msa", "كم سعر صهريج المياه الخاص؟", None, None),
    ("q15", "gulf", "ممكن ادفع الفاتوره كاش في محطة البنزين؟", None, None),
    ("q16", "en", "Do you refund the cost of a solar water heater?", None, None),
]

REPLY_TASKS = [
    ("r01", "leaks", "There is a big water leak in the street outside my house in Al-Rabwa", {"ticket": "SW-48211", "crew_eta_hours": 4}),
    ("r02", "supply", "الماي مقطوع عندنا في النسيم من الصبح", {"ticket": "SW-48212", "planned_interruption": "no"}),
    ("r03", "billing", "I got a bill for 1340 SAR and I don't understand why it is so high", {"ticket": "SW-48213", "last_month_bill_sar": 410}),
    ("r04", "meters", "ابي تفحصون العداد يحسب غلط", {"ticket": "SW-48214", "test_fee_sar": 150}),
    ("r05", "payments", "I would like to pay 2210 SAR in monthly instalments", {"ticket": "SW-48215", "max_instalments": 6}),
    ("r06", "quality", "The water coming out of the tap is brown", {"ticket": "SW-48216", "lab_visit_days": 2}),
    ("r07", "wastewater", "المجاري طافحه بالشارع في الغدير", {"ticket": "SW-48217", "crew_eta_hours": 3}),
    ("r08", "connections", "I bought the house and want the account moved to my name", {"ticket": "SW-48218", "fee_sar": 0}),
    ("r09", "supply", "el mai maqtoo3 3indna fi el manhal min el sob7", {"ticket": "SW-48219", "planned_interruption": "yes"}),
    ("r10", "billing", "I was charged twice for the same bill", {"ticket": "SW-48220", "refund_days": 7}),
]

SMOKE_EDGE = [
    ("E01", "", None, "an empty message"),
    ("E02", "؟؟؟", None, "punctuation only"),
    ("E03", "hello", None, "a greeting with no request"),
    ("E04", "الماي مقطوع والمجاري طافحه والفاتوره غاليه", None, "three problems in one message"),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    DATA.mkdir(exist_ok=True)
    (DATA / "rules").mkdir(exist_ok=True)
    customers = make_customers(rng)
    messages = make_messages(customers, rng)
    split = split_by_customer(customers, messages, rng)
    write_csv(DATA / "customers.csv", customers)
    cols = ["msg_id", "customer_id", "variety", "text", "department", "complaint_type", "account_number", "district", "has_phone", "has_name", "template"]
    write_csv(DATA / "messages.csv", [{k: ("" if m[k] is None else m[k]) for k in cols} for m in messages])
    write_csv(DATA / "split.csv", [{"customer_id": c, "split": s} for c, s in sorted(split.items())])

    # 50 parallel messages: every template once, the same facts in all four varieties
    prng = random.Random(SEED + 1)
    parallel = []
    for i, (ctype, *_) in enumerate(TEMPLATES):
        cust = prng.choice(customers)
        slots = random_slots(cust, prng)
        row = {"id": f"P{i + 1:02d}", "complaint_type": ctype}
        for v in VARIETIES:
            row[v] = fill(TEMPLATES[i][1 + VARIETIES.index(v)], v, slots)
        parallel.append(row)
    write_csv(DATA / "parallel_50.csv", parallel)

    # 40 queries: 10 per variety, each a (complaint type, district) pair with 3 to 20 relevant messages
    cells = {}
    for m in messages:
        if m["district"]:
            cells.setdefault((m["complaint_type"], m["district"]), []).append(m["msg_id"])
    good = sorted(k for k, v in cells.items() if 3 <= len(v) <= 20)
    qrng = random.Random(SEED + 2)
    qrng.shuffle(good)
    queries = []
    for j in range(10):  # interleaved, so that the first 20 hold 5 per variety
        for vi, v in enumerate(VARIETIES):
            ctype, dkey = good[vi * 10 + j]
            phrase = QUERY_PHRASES[ctype][vi]
            joiner = {"msa": " في حي ", "gulf": " في ", "arabizi": " fi ", "en": " in "}[v]
            queries.append({"qid": f"Q{len(queries) + 1:02d}", "variety": v, "query": phrase + joiner + dist_text(dkey, v),
                            "complaint_type": ctype, "district": dkey, "relevant": cells[(ctype, dkey)]})
    write_jsonl(DATA / "queries.jsonl", queries)

    test_ids = [m for m in messages if split[m["customer_id"]] == "test"]
    xrng = random.Random(SEED + 3)
    sample = sorted(xrng.sample(test_ids, 200), key=lambda m: m["msg_id"])
    write_jsonl(DATA / "extraction_200.jsonl", [{"msg_id": m["msg_id"], "variety": m["variety"], "text": m["text"],
                                                "gold": {"account_number": m["account_number"], "district": m["district"], "complaint_type": m["complaint_type"]}}
                                               for m in sample])

    srng = random.Random(SEED + 4)
    smoke = []
    for v in VARIETIES:
        pool = [m for m in test_ids if m["variety"] == v]
        for m in sorted(srng.sample(pool, 5), key=lambda m: m["msg_id"]):
            smoke.append({"id": m["msg_id"], "text": m["text"], "variety": v, "department": m["department"], "handoff": False})
    for sid, text, dept, why in SMOKE_EDGE:
        smoke.append({"id": sid, "text": text, "variety": "edge", "department": dept, "handoff": True, "note": why})
    write_jsonl(DATA / "smoke_test.jsonl", smoke)

    write_jsonl(DATA / "reply_tasks.jsonl", [{"id": i, "department": d, "message": t, "facts": f} for i, d, t, f in REPLY_TASKS])
    for rid, en_title, ar_title, en, ar in RULES:
        slug = en_title.lower().replace(",", "").replace(" ", "-")
        (DATA / "rules" / f"{rid}_{slug}.md").write_text(f"# {rid} · {en_title}\n\n{en}\n\n## {ar_title}\n\n{ar}\n", encoding="utf-8", newline="\n")
    write_jsonl(DATA / "rule_questions.jsonl", [{"qid": q, "variety": v, "question": t, "rule": r, "key_fact": k} for q, v, t, r, k in RULE_QUESTIONS])

    by_split = {}
    for m in messages:
        by_split[split[m["customer_id"]]] = by_split.get(split[m["customer_id"]], 0) + 1
    print(f"customers {len(customers)}, messages {len(messages)} {by_split}, queries {len(queries)}, rules {len(RULES)}")


if __name__ == "__main__":
    main()
