import streamlit as st
import json
from google.oauth2 import id_token
from google.auth.transport import requests
import os
from typing import Dict, Any
from general_user import general_user_dashboard as music_therapy_dashboard
from caregiver import caregiver_dashboard as ml_caregiver_dashboard

# Configure Streamlit page
st.set_page_config(
    page_title="Healthcare Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
CAREGIVER_EMAILS = [
    "vani.kandasamy@pyxeda.ai"
    # Add more caregiver email addresses here
]

# Google OAuth configuration
GOOGLE_CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]

def verify_google_token(token: str) -> Dict[str, Any]:
    """Verify Google ID token and return user info"""
    try:
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), GOOGLE_CLIENT_ID
        )
        return idinfo
    except ValueError:
        return None

def is_caregiver(email: str) -> bool:
    """Check if email belongs to a caregiver"""
    return email.lower() in [email.lower() for email in CAREGIVER_EMAILS]

def login_page():
    """Display login page with Google Sign-In"""
    st.title("🏥 Healthcare Portal")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3>Welcome to Healthcare Portal</h3>
            <p>Please sign in with your Google account to continue</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Google Sign-In button (placeholder - requires frontend integration)
        st.markdown("""
        <div style='text-align: center; margin: 2rem 0;'>
            <div id="g_id_onload"
                 data-client_id="your-google-client-id"
                 data-callback="handleCredentialResponse">
            </div>
            <div class="g_id_signin" data-type="standard"></div>
        </div>
        
        <script src="https://accounts.google.com/gsi/client" async defer></script>
        <script>
        function handleCredentialResponse(response) {
            // This would normally send the token to Streamlit
            console.log("Encoded JWT ID token: " + response.credential);
        }
        </script>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("📝 **For Demo**: Enter your email below to simulate login")
        
        # Demo login form
        with st.form("demo_login"):
            email = st.text_input("Email Address", placeholder="user@example.com")
            name = st.text_input("Full Name", placeholder="John Doe")
            submit = st.form_submit_button("Sign In (Demo)", use_container_width=True)
            
            if submit and email and name:
                # Store user info in session state
                st.session_state.user_info = {
                    'email': email,
                    'name': name,
                    'verified_email': True
                }
                st.session_state.authenticated = True
                st.rerun()



def main():
    """Main application logic"""
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        login_page()
    else:
        user_email = st.session_state.user_info['email']
        
        if is_caregiver(user_email):
            ml_caregiver_dashboard()
        else:
            music_therapy_dashboard()

if __name__ == "__main__":
    main()