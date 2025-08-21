import streamlit as st
from typing import Dict, Any, Optional
from general_user import general_user_dashboard as music_therapy_dashboard
from caregiver import caregiver_dashboard as ml_caregiver_dashboard

# Configure Streamlit page
st.set_page_config(
    page_title="NeuroTunes",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMAGE_ADDRESS = "https://www.denvercenter.org/wp-content/uploads/2024/10/music-therapy.jpg"

# Caregiver emails (simple list)
CAREGIVER_EMAILS = [
    "aiclubcolab@gmail.com"
]

def is_caregiver(email: str) -> bool:
    return email.lower() in [e.lower() for e in CAREGIVER_EMAILS]

def get_user_simple() -> Optional[Dict[str, Any]]:
    """Use Streamlit experimental auth only (no fallback)."""
    exp_user = getattr(st, "experimental_user", None)
    has_login = hasattr(st, "login") and hasattr(st, "logout")
    if not (exp_user is not None and hasattr(exp_user, "is_logged_in") and has_login):
        st.error("Streamlit experimental auth is not available in this environment. Please upgrade Streamlit or enable authentication.")
        st.stop()
    # Sidebar auth controls
    with st.sidebar:
        if not exp_user.is_logged_in:
            if st.button("Log in with Google", type="primary"):
                st.login()
            return None
        else:
            if st.button("Log out", type="secondary"):
                st.logout()
                return None
    # Return user info
    name = getattr(exp_user, "name", None) or getattr(exp_user, "username", None) or "User"
    email = getattr(exp_user, "email", None) or ""
    st.markdown(f"Hello, <span style='color: orange; font-weight: bold;'>{name}</span>!", unsafe_allow_html=True)
    return {"name": name, "email": email}



def main():
    """Main application logic (simple auth + role routing)."""
    # Title and image
    st.title("NeuroTunes")
    st.image(IMAGE_ADDRESS, caption="EEG Frequency Bands (Delta, Theta, Alpha, Beta, Gamma)")
    st.markdown("---")

    # Authenticate
    user = get_user_simple()
    if not user:
        st.stop()
    st.session_state.user_info = user

    # Route by role based on email
    user_email = (user.get("email") or "").strip()
    if user_email and is_caregiver(user_email):
        # Caregiver dashboard
        ml_caregiver_dashboard()
    else:
        # General user dashboard
        music_therapy_dashboard()

if __name__ == "__main__":
    main()