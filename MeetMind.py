
import os
import io
import re
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil import parser

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="MeetMind",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
}

section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #f8fafc;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 25px;
}

.metric-card {
    background: #111827;
    border: 1px solid #263247;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
}

.feature-card {
    background: #111827;
    border: 1px solid #263247;
    border-radius: 16px;
    padding: 22px;
    min-height: 160px;
}

.section-title {
    color: white;
    font-size: 23px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.success-box {
    background: #052e1b;
    border: 1px solid #047857;
    border-radius: 12px;
    padding: 15px;
    color: #6ee7b7;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "action_items" not in st.session_state:
    st.session_state.action_items = []

if "meeting_history" not in st.session_state:
    st.session_state.meeting_history = []

if "transcript" not in st.session_state:
    st.session_state.transcript = ""


# ============================================================
# AUDIO TRANSCRIPTION
# ============================================================

def get_openai_key():
    key = os.getenv("OPENAI_API_KEY")

    if key:
        return key

    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


def transcribe_audio(audio_bytes, filename, content_type):
    if OpenAI is None:
        st.error("OpenAI package is not installed. Run: pip install openai")
        return None

    api_key = get_openai_key()

    if not api_key:
        st.error("OPENAI_API_KEY is not configured.")
        return None

    try:
        client = OpenAI(api_key=api_key)

        audio_file = (
            filename,
            io.BytesIO(audio_bytes),
            content_type
        )

        result = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file
        )

        return result.text

    except Exception as e:
        st.error(f"Audio transcription failed: {e}")
        return None


# ============================================================
# TEXT FUNCTIONS
# ============================================================

def clean_text(text):
    text = str(text).replace("\r", "")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(text):
    text = clean_text(text)
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?])\s+|\n+", text)
        if x.strip()
    ]


def get_speaker(sentence):
    match = re.match(
        r"^([A-Za-z][A-Za-z0-9 _-]{0,30}):",
        sentence
    )

    if match:
        return match.group(1).strip()

    return None


def extract_owner(sentence):
    speaker = get_speaker(sentence)

    if speaker:
        return speaker

    patterns = [
        r"\b([A-Z][a-z]+)\s+will\b",
        r"\b([A-Z][a-z]+)\s+should\b",
        r"\b([A-Z][a-z]+)\s+needs\s+to\b",
        r"\b([A-Z][a-z]+)\s+has\s+to\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, sentence)

        if match:
            return match.group(1)

    return None


def extract_task(sentence):
    text = re.sub(
        r"^[A-Za-z][A-Za-z0-9 _-]{0,30}:\s*",
        "",
        sentence
    ).strip()

    patterns = [
        r"\bI will\s+(.+)",
        r"\bI'll\s+(.+)",
        r"\bI need to\s+(.+)",
        r"\bI should\s+(.+)",
        r"\bI have to\s+(.+)",
        r"\bI have completed\s+(.+)",
        r"\bI have finished\s+(.+)",
        r"\bI am\s+(?:currently\s+)?working on\s+(.+)",
        r"\bI am\s+(?:currently\s+)?testing\s+(.+)",
        r"\bI'm\s+(?:currently\s+)?working on\s+(.+)",
        r"\bI'm\s+(?:currently\s+)?testing\s+(.+)",
        r"\bwill\s+(.+)",
        r"\bshould\s+(.+)",
        r"\bneeds to\s+(.+)",
        r"\bhas to\s+(.+)",
        r"\bmust\s+(.+)"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            task = match.group(1)

            task = re.sub(
                r"\s+(?:by|before|on)\s+.+$",
                "",
                task,
                flags=re.IGNORECASE
            )

            task = task.strip(" .,")

            if task:
                return task.capitalize()

    # Status-based sentences
    cleaned = text

    cleaned = re.sub(
        r"\s+(?:is|are)\s+(?:still\s+)?remaining\.?$",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s+(?:is|are|has been)\s+(?:almost\s+|nearly\s+)?(?:completed|complete|finished|done)\.?$",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    status_words = [
        "completed",
        "complete",
        "finished",
        "done",
        "working",
        "testing",
        "in progress",
        "developing",
        "implementing",
        "started",
        "ongoing",
        "pending",
        "remaining"
    ]

    if any(word in text.lower() for word in status_words):
        return cleaned.strip(" .,")

    return None


def extract_deadline(sentence):
    text = sentence.lower()
    today = datetime.now().date()

    if "today" in text:
        return str(today)

    if "tomorrow" in text:
        return str(today + timedelta(days=1))

    # Example: 29 September
    match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\b",
        sentence,
        re.IGNORECASE
    )

    if match:
        try:
            day = int(match.group(1))
            month = datetime.strptime(
                match.group(2),
                "%B"
            ).month

            year = today.year
            result = datetime(year, month, day).date()

            if result < today:
                result = datetime(year + 1, month, day).date()

            return str(result)

        except ValueError:
            pass

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    for day, number in weekdays.items():
        if day in text:
            days = (number - today.weekday()) % 7

            if days == 0:
                days = 7

            return str(today + timedelta(days=days))

    patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, sentence)

        if match:
            try:
                value = parser.parse(
                    match.group(0),
                    dayfirst=True
                )

                return str(value.date())

            except ValueError:
                pass

    return None


# ============================================================
# STATUS
# ============================================================

def detect_status(sentence):
    text = sentence.lower()

    if any(x in text for x in [
        "still remaining",
        "is remaining",
        "are remaining",
        "pending",
        "not started",
        "yet to start"
    ]):
        return "Pending"

    if any(x in text for x in [
        "currently working",
        "working on",
        "currently testing",
        "testing",
        "in progress",
        "developing",
        "implementing",
        "started",
        "ongoing",
        "almost completed",
        "almost complete",
        "nearly completed",
        "nearly complete"
    ]):
        return "In Progress"

    if any(x in text for x in [
        "have completed",
        "has completed",
        "completed",
        "complete",
        "finished",
        "done"
    ]):
        return "Completed"

    return "Pending"


def calculate_confidence(task, owner, deadline):
    score = 0.60

    if task:
        score += 0.15

    if owner:
        score += 0.10

    if deadline:
        score += 0.10

    return min(score, 0.95)


def calculate_risk(deadline, status):
    if not deadline:
        return "Unknown"

    try:
        date_value = datetime.strptime(
            deadline,
            "%Y-%m-%d"
        ).date()

        days_left = (
            date_value - datetime.now().date()
        ).days

        if status == "Completed":
            return "Low"

        if days_left <= 1:
            return "High"

        if days_left <= 3:
            return "Medium"

        return "Low"

    except ValueError:
        return "Unknown"


def extract_action_items(transcript):
    results = []

    for sentence in split_sentences(transcript):

        task = extract_task(sentence)

        if not task:
            continue

        owner = extract_owner(sentence)
        deadline = extract_deadline(sentence)
        status = detect_status(sentence)

        confidence = calculate_confidence(
            task,
            owner,
            deadline
        )

        reasons = []

        if not owner:
            reasons.append("Owner is unclear")

        if not deadline:
            reasons.append("Deadline is missing")

        unclear = len(reasons) > 0

        duplicate = any(
            item["Task"].lower() == task.lower()
            for item in results
        )

        results.append({
            "Task": task,
            "Owner": owner or "Unknown",
            "Deadline": deadline or "Not specified",
            "Status": status,
            "Confidence": round(confidence * 100),
            "Deadline Risk": calculate_risk(
                deadline,
                status
            ),
            "Unclear": "Yes" if unclear else "No",
            "Reason": ", ".join(reasons),
            "Duplicate": "Yes" if duplicate else "No"
        })

    return results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center">
            <div style="font-size:48px;">🧠</div>
            <div style="font-size:23px;font-weight:800;">
                MeetMind AI
            </div>
            <div style="color:#64748b;font-size:12px;">
                Intelligent Meeting Assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Overview",
            "📄 Extract Actions",
            "📊 Analytics",
            "👥 Team Workload",
            "⚠️ Attention Required",
            "📜 Meeting History"
        ]
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">MeetMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Turn meeting conversations into clear, trackable actions.</div>',
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    items = st.session_state.action_items

    total = len(items)

    pending = sum(
        x["Status"] == "Pending"
        for x in items
    )

    high_risk = sum(
        x["Deadline Risk"] == "High"
        for x in items
    )

    unclear = sum(
        x["Unclear"] == "Yes"
        for x in items
    )

    columns = st.columns(4)

    values = [
        ("📋", "Total Tasks", total),
        ("⏳", "Pending", pending),
        ("🚨", "High Risk", high_risk),
        ("⚠️", "Needs Attention", unclear)
    ]

    for col, value in zip(columns, values):

        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:25px;">{value[0]}</div>
                    <div style="color:#94a3b8;">{value[1]}</div>
                    <div style="color:white;font-size:32px;font-weight:800;">
                        {value[2]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">What MeetMind can do</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    features = [
        ("🤖", "AI Action Extraction",
         "Find tasks, owners, deadlines and status."),
        ("⏰", "Deadline Intelligence",
         "Detect deadlines and deadline risks."),
        ("👥", "Team Workload",
         "Understand task distribution."),
        ("🗣️", "Unclear Actions",
         "Find missing owners or deadlines."),
        ("🔍", "Duplicate Detection",
         "Find repeated action items."),
        ("📥", "Easy Export",
         "Download action items as CSV.")
    ]

    for i, feature in enumerate(features):

        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div style="font-size:30px;">{feature[0]}</div>
                    <h3 style="color:white;">{feature[1]}</h3>
                    <p style="color:#94a3b8;">
                        {feature[2]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# EXTRACT ACTIONS
# ============================================================

elif page == "📄 Extract Actions":

    st.markdown(
        '<div class="section-title">📄 Analyze Meeting Transcript</div>',
        unsafe_allow_html=True
    )

    # TXT FILE
    txt_file = st.file_uploader(
        "Upload TXT transcript",
        type=["txt"],
        key="txt_upload"
    )

    if txt_file:
        st.session_state.transcript = txt_file.read().decode(
            "utf-8",
            errors="replace"
        )

    # AUDIO
    st.markdown("### 🎙️ Upload Meeting Audio")

    audio_file = st.file_uploader(
        "Upload audio",
        type=[
            "mp3",
            "mp4",
            "mpeg",
            "mpga",
            "m4a",
            "wav",
            "webm",
            "ogg"
        ],
        key="audio_upload"
    )

    if audio_file:

        st.audio(audio_file)

        if st.button(
            "🎙️ Convert Audio to Text",
            use_container_width=True
        ):

            text = transcribe_audio(
                audio_file.getvalue(),
                audio_file.name,
                audio_file.type
            )

            if text:
                st.session_state.transcript = text
                st.success("Audio converted successfully.")

    # MICROPHONE
    st.markdown("### 🎤 Record from Microphone")

    mic_audio = st.audio_input(
        "Record your meeting"
    )

    if mic_audio:

        if st.button(
            "📝 Convert Recording to Text",
            use_container_width=True
        ):

            text = transcribe_audio(
                mic_audio.getvalue(),
                "microphone.wav",
                "audio/wav"
            )

            if text:
                st.session_state.transcript = text
                st.success("Recording converted successfully.")

    # TRANSCRIPT
    transcript = st.text_area(
        "Meeting Transcript",
        value=st.session_state.transcript,
        height=280,
        placeholder="""Example:

Rahul: I have completed the frontend dashboard.
Priya: I am currently working on the database.
Amit: I will complete API testing by 3 October."""
    )

    st.session_state.transcript = transcript

    if st.button(
        "✨ Extract Action Items",
        type="primary",
        use_container_width=True
    ):

        if not transcript.strip():

            st.warning("Please provide a meeting transcript.")

        else:

            results = extract_action_items(
                transcript
            )

            st.session_state.action_items = results

            st.session_state.meeting_history.append({
                "Meeting":
                    f"Meeting {len(st.session_state.meeting_history) + 1}",
                "Date":
                    str(datetime.now().date()),
                "Action Items":
                    len(results)
            })

            st.success(
                f"Analysis complete — {len(results)} action items found."
            )

    # RESULTS
    if st.session_state.action_items:

        st.markdown(
            '<div class="section-title">📋 Extracted Action Items</div>',
            unsafe_allow_html=True
        )

        df = pd.DataFrame(
            st.session_state.action_items
        )

        st.dataframe(
            df[
                [
                    "Task",
                    "Owner",
                    "Deadline",
                    "Status",
                    "Confidence",
                    "Deadline Risk",
                    "Unclear"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Action Items",
            df.to_csv(index=False).encode("utf-8"),
            "meeting_action_items.csv",
            "text/csv",
            use_container_width=True
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="main-title">Team Task Completion Overview</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.action_items:

        st.info("Please extract action items first.")

    else:

        df = pd.DataFrame(
            st.session_state.action_items
        )

        team_data = {}

        for _, row in df.iterrows():

            owner = str(
                row.get("Owner", "Unknown")
            ).strip()

            status = str(
                row.get("Status", "Pending")
            ).lower().strip()

            if not owner or owner.lower() in ["none", "nan"]:
                owner = "Unknown"

            if owner not in team_data:
                team_data[owner] = {
                    "Completed": 0,
                    "In Progress": 0,
                    "Remaining": 0
                }

            if status == "completed":
                team_data[owner]["Completed"] += 1

            elif status == "in progress":
                team_data[owner]["In Progress"] += 1

            else:
                team_data[owner]["Remaining"] += 1

        names = list(team_data.keys())

        completed = []
        progress = []
        remaining = []

        for name in names:

            data = team_data[name]

            total = (
                data["Completed"]
                + data["In Progress"]
                + data["Remaining"]
            )

            completed.append(
                data["Completed"] / total * 100
            )

            progress.append(
                data["In Progress"] / total * 100
            )

            remaining.append(
                data["Remaining"] / total * 100
            )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=names,
                y=completed,
                name="Completed",
                text=[f"{x:.1f}%" for x in completed],
                textposition="inside",
                marker_color="#16C7A5"
            )
        )

        fig.add_trace(
            go.Bar(
                x=names,
                y=progress,
                name="In Progress",
                text=[f"{x:.1f}%" for x in progress],
                textposition="inside",
                marker_color="#3498DB"
            )
        )

        fig.add_trace(
            go.Bar(
                x=names,
                y=remaining,
                name="Remaining",
                text=[f"{x:.1f}%" for x in remaining],
                textposition="inside",
                marker_color="#E91E63"
            )
        )

        fig.update_layout(
            barmode="stack",
            height=600,
            yaxis={
                "range": [0, 100],
                "ticksuffix": "%"
            },
            plot_bgcolor="#081225",
            paper_bgcolor="#081225",
            font={"color": "white"},
            legend={
                "orientation": "h",
                "y": 1.02,
                "x": 0.5,
                "xanchor": "center"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        completed_count = sum(
            x["Status"] == "Completed"
            for x in st.session_state.action_items
        )

        progress_count = sum(
            x["Status"] == "In Progress"
            for x in st.session_state.action_items
        )

        remaining_count = sum(
            x["Status"] == "Pending"
            for x in st.session_state.action_items
        )

        total = (
            completed_count
            + progress_count
            + remaining_count
        )

        completed_percentage = (
            completed_count / total * 100
            if total else 0
        )

        progress_percentage = (
            progress_count / total * 100
            if total else 0
        )

        remaining_percentage = (
            remaining_count / total * 100
            if total else 0
        )

        st.markdown(
            '<div class="section-title">Overall Team Progress</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        cards = [
            ("🟢", "Completed", completed_percentage, completed_count),
            ("🔵", "Completing Now", progress_percentage, progress_count),
            ("🔴", "Remaining", remaining_percentage, remaining_count),
            ("📋", "Total Tasks", total, total)
        ]

        for col, card in zip(
            [c1, c2, c3, c4],
            cards
        ):

            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size:25px;">{card[0]}</div>
                        <div style="color:#94a3b8;">
                            {card[1]}
                        </div>
                        <div style="color:white;font-size:30px;font-weight:800;">
                            {card[2]:.1f}%
                        </div>
                        <div style="color:#94a3b8;">
                            {card[3]} task(s)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# TEAM WORKLOAD
# ============================================================

elif page == "👥 Team Workload":

    st.markdown(
        '<div class="main-title">Team Workload</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.action_items:

        st.info("Please extract action items first.")

    else:

        df = pd.DataFrame(
            st.session_state.action_items
        )

        rows = []

        for owner in df["Owner"].fillna("Unknown").unique():

            owner_df = df[
                df["Owner"].fillna("Unknown") == owner
            ]

            total = len(owner_df)

            completed = sum(
                owner_df["Status"] == "Completed"
            )

            progress = sum(
                owner_df["Status"] == "In Progress"
            )

            remaining = sum(
                owner_df["Status"] == "Pending"
            )

            high_risk = sum(
                owner_df["Deadline Risk"] == "High"
            )

            confidence = pd.to_numeric(
                owner_df["Confidence"],
                errors="coerce"
            ).mean()

            completion = (
                completed / total * 100
                if total else 0
            )

            if total >= 6:
                workload = "🔴 High"
            elif total >= 3:
                workload = "🟡 Medium"
            else:
                workload = "🟢 Low"

            rows.append({
                "Team Member": owner,
                "Total Tasks": total,
                "Completed": completed,
                "In Progress": progress,
                "Remaining": remaining,
                "High Risk": high_risk,
                "Avg Confidence": confidence if pd.notna(confidence) else 0,
                "Completion": completion,
                "Workload": workload
            })

        workload_df = pd.DataFrame(rows)

        st.dataframe(
            workload_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ATTENTION REQUIRED
# ============================================================

elif page == "⚠️ Attention Required":

    st.markdown(
        '<div class="section-title">⚠️ Tasks Requiring Attention</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.action_items:

        st.info("Analyze a transcript first.")

    else:

        df = pd.DataFrame(
            st.session_state.action_items
        )

        unclear = df[
            df["Unclear"] == "Yes"
        ]

        high_risk = df[
            df["Deadline Risk"] == "High"
        ]

        st.markdown("### 🗣️ Unclear Actions")

        if unclear.empty:

            st.markdown(
                '<div class="success-box">✓ No unclear actions detected.</div>',
                unsafe_allow_html=True
            )

        else:

            st.dataframe(
                unclear[
                    [
                        "Task",
                        "Owner",
                        "Deadline",
                        "Reason"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        st.markdown("### 🚨 High-Risk Deadlines")

        if high_risk.empty:

            st.markdown(
                '<div class="success-box">✓ No high-risk deadlines detected.</div>',
                unsafe_allow_html=True
            )

        else:

            st.dataframe(
                high_risk[
                    [
                        "Task",
                        "Owner",
                        "Deadline",
                        "Status"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# MEETING HISTORY
# ============================================================

elif page == "📜 Meeting History":

    st.markdown(
        '<div class="section-title">📜 Meeting History</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.meeting_history:

        st.info("No meetings have been analyzed yet.")

    else:

        st.dataframe(
            pd.DataFrame(
                st.session_state.meeting_history
            ),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>
    <div style="
        text-align:center;
        color:#64748b;
        padding:20px;
        border-top:1px solid #1f2937;
    ">
        <b style="color:#94a3b8;">MeetMind AI</b>
        &nbsp; • &nbsp;
        AI Meeting Action-Item Extractor
        <br>
        <small>Python • Streamlit • NLP • Data Analytics</small>
    </div>
    """,
    unsafe_allow_html=True
)
