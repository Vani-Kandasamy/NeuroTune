import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import json

# Melody database with 5 music genres
MELODY_DATABASE = {
    "Classical": [
        {"id": 1, "name": "Bach's Prelude", "duration": 240, "bpm": 72, "key": "C Major", "neural_score": 9.2},
        {"id": 2, "name": "Mozart's Sonata", "duration": 280, "bpm": 68, "key": "G Major", "neural_score": 9.0},
        {"id": 3, "name": "Beethoven's Symphony", "duration": 320, "bpm": 76, "key": "F Major", "neural_score": 9.5},
        {"id": 4, "name": "Chopin's Nocturne", "duration": 200, "bpm": 65, "key": "D Major", "neural_score": 8.8},
        {"id": 5, "name": "Vivaldi's Spring", "duration": 260, "bpm": 80, "key": "A Major", "neural_score": 9.1},
        {"id": 6, "name": "Debussy's Clair de Lune", "duration": 220, "bpm": 62, "key": "E Major", "neural_score": 8.9},
        {"id": 7, "name": "Pachelbel's Canon", "duration": 300, "bpm": 70, "key": "B♭ Major", "neural_score": 9.3},
        {"id": 8, "name": "Schubert's Ave Maria", "duration": 180, "bpm": 60, "key": "C Major", "neural_score": 8.7},
        {"id": 9, "name": "Brahms' Lullaby", "duration": 160, "bpm": 58, "key": "G Major", "neural_score": 8.5}
    ],
    "Rock": [
        {"id": 10, "name": "Thunder Strike", "duration": 210, "bpm": 140, "key": "E Minor", "neural_score": 8.4},
        {"id": 11, "name": "Electric Storm", "duration": 195, "bpm": 145, "key": "A Minor", "neural_score": 8.7},
        {"id": 12, "name": "Power Chord", "duration": 180, "bpm": 135, "key": "D Minor", "neural_score": 8.2},
        {"id": 13, "name": "Rock Anthem", "duration": 240, "bpm": 130, "key": "G Minor", "neural_score": 8.9},
        {"id": 14, "name": "Guitar Hero", "duration": 220, "bpm": 138, "key": "C Minor", "neural_score": 8.6},
        {"id": 15, "name": "Metal Fusion", "duration": 200, "bpm": 142, "key": "F Minor", "neural_score": 8.3},
        {"id": 16, "name": "Drum Solo", "duration": 160, "bpm": 150, "key": "B Minor", "neural_score": 8.1},
        {"id": 17, "name": "Bass Drop", "duration": 185, "bpm": 136, "key": "E Minor", "neural_score": 8.5},
        {"id": 18, "name": "Amplified", "duration": 205, "bpm": 144, "key": "A Minor", "neural_score": 8.8}
    ],
    "Pop": [
        {"id": 19, "name": "Catchy Beat", "duration": 180, "bpm": 120, "key": "C Major", "neural_score": 8.0},
        {"id": 20, "name": "Dance Floor", "duration": 200, "bpm": 125, "key": "G Major", "neural_score": 8.3},
        {"id": 21, "name": "Radio Hit", "duration": 190, "bpm": 118, "key": "F Major", "neural_score": 7.9},
        {"id": 22, "name": "Upbeat Melody", "duration": 175, "bpm": 122, "key": "D Major", "neural_score": 8.2},
        {"id": 23, "name": "Feel Good", "duration": 185, "bpm": 115, "key": "A Major", "neural_score": 8.1},
        {"id": 24, "name": "Summer Vibes", "duration": 195, "bpm": 128, "key": "E Major", "neural_score": 8.4},
        {"id": 25, "name": "Chart Topper", "duration": 170, "bpm": 120, "key": "B♭ Major", "neural_score": 7.8},
        {"id": 26, "name": "Mainstream", "duration": 188, "bpm": 124, "key": "C Major", "neural_score": 8.0},
        {"id": 27, "name": "Pop Anthem", "duration": 205, "bpm": 116, "key": "G Major", "neural_score": 8.3}
    ],
    "Rap": [
        {"id": 28, "name": "Street Beats", "duration": 200, "bpm": 95, "key": "E Minor", "neural_score": 7.8},
        {"id": 29, "name": "Urban Flow", "duration": 180, "bpm": 88, "key": "A Minor", "neural_score": 8.1},
        {"id": 30, "name": "Hip Hop Classic", "duration": 220, "bpm": 92, "key": "D Minor", "neural_score": 8.4},
        {"id": 31, "name": "Freestyle", "duration": 160, "bpm": 100, "key": "G Minor", "neural_score": 7.6},
        {"id": 32, "name": "Boom Bap", "duration": 195, "bpm": 85, "key": "C Minor", "neural_score": 8.0},
        {"id": 33, "name": "Trap Beat", "duration": 175, "bpm": 105, "key": "F Minor", "neural_score": 7.9},
        {"id": 34, "name": "Conscious Rap", "duration": 240, "bpm": 90, "key": "B Minor", "neural_score": 8.3},
        {"id": 35, "name": "Underground", "duration": 210, "bpm": 87, "key": "E Minor", "neural_score": 8.2},
        {"id": 36, "name": "Lyrical Flow", "duration": 185, "bpm": 93, "key": "A Minor", "neural_score": 7.7}
    ],
    "R&B": [
        {"id": 37, "name": "Smooth Soul", "duration": 220, "bpm": 75, "key": "C Major", "neural_score": 8.6},
        {"id": 38, "name": "Velvet Voice", "duration": 240, "bpm": 70, "key": "G Major", "neural_score": 8.8},
        {"id": 39, "name": "Groove Master", "duration": 200, "bpm": 78, "key": "F Major", "neural_score": 8.4},
        {"id": 40, "name": "Soulful Nights", "duration": 260, "bpm": 68, "key": "D Major", "neural_score": 8.9},
        {"id": 41, "name": "Love Ballad", "duration": 210, "bpm": 72, "key": "A Major", "neural_score": 8.7},
        {"id": 42, "name": "Midnight Groove", "duration": 195, "bpm": 76, "key": "E Major", "neural_score": 8.3},
        {"id": 43, "name": "Neo Soul", "duration": 225, "bpm": 74, "key": "B♭ Major", "neural_score": 8.6},
        {"id": 44, "name": "Rhythm & Blues", "duration": 250, "bpm": 65, "key": "C Major", "neural_score": 8.8},
        {"id": 45, "name": "Smooth Operator", "duration": 180, "bpm": 80, "key": "G Major", "neural_score": 8.1}
    ]
}

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_track' not in st.session_state:
        st.session_state.current_track = None
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'playback_position' not in st.session_state:
        st.session_state.playback_position = 0
    if 'listening_history' not in st.session_state:
        st.session_state.listening_history = []
    if 'neural_data' not in st.session_state:
        st.session_state.neural_data = generate_sample_neural_data()
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {"volume": 50, "preferred_categories": ["Classical"]}

def generate_sample_neural_data():
    """Generate sample neural engagement data"""
    data = []
    for category, tracks in MELODY_DATABASE.items():
        for track in tracks:
            # Simulate listening sessions over the past month
            for _ in range(random.randint(1, 5)):
                date = datetime.now() - timedelta(days=random.randint(0, 30))
                engagement_score = track['neural_score'] + random.uniform(-1, 1)
                data.append({
                    'date': date,
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'category': category,
                    'neural_engagement': max(0, min(10, engagement_score)),
                    'duration_listened': random.randint(30, track['duration']),
                    'completion_rate': random.uniform(0.3, 1.0)
                })
    return pd.DataFrame(data)

def music_player_widget(track):
    """Create a music player widget"""
    if track:
        col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
        
        with col1:
            if st.button("⏯️", key=f"play_{track['id']}"):
                st.session_state.is_playing = not st.session_state.is_playing
        
        with col2:
            st.write(f"**{track['name']}**")
            progress = st.session_state.playback_position / track['duration']
            st.progress(progress)
        
        with col3:
            st.write(f"{st.session_state.playback_position//60:02d}:{st.session_state.playback_position%60:02d}")
        
        with col4:
            st.write(f"{track['duration']//60:02d}:{track['duration']%60:02d}")
        
        # Volume control
        volume = st.slider("🔊 Volume", 0, 100, st.session_state.user_preferences["volume"], key="volume_slider")
        st.session_state.user_preferences["volume"] = volume
        
        # Simulate playback
        if st.session_state.is_playing and st.session_state.playback_position < track['duration']:
            time.sleep(0.1)  # Simulate real-time playback
            st.session_state.playback_position += 1
            st.rerun()
        elif st.session_state.playback_position >= track['duration']:
            st.session_state.is_playing = False
            st.session_state.playback_position = 0

def track_card(track, category):
    """Create a track card with play button and details"""
    with st.container():
        col1, col2, col3 = st.columns([1, 4, 1])
        
        with col1:
            if st.button("▶️", key=f"card_play_{track['id']}"):
                st.session_state.current_track = track
                st.session_state.is_playing = True
                st.session_state.playback_position = 0
                
                # Add to listening history
                st.session_state.listening_history.append({
                    'timestamp': datetime.now(),
                    'track': track,
                    'category': category
                })
        
        with col2:
            st.markdown(f"""
            **{track['name']}**  
            Category: {category} | Duration: {track['duration']//60}:{track['duration']%60:02d} | BPM: {track['bpm']}  
            Key: {track['key']} | Neural Score: ⭐ {track['neural_score']}/10
            """)
        
        with col3:
            st.metric("Score", f"{track['neural_score']:.1f}")

def generate_playlist_recommendations(top_track):
    """Generate playlist based on highest engaged track"""
    recommendations = []
    
    # Find tracks with similar neural scores and characteristics
    for category, tracks in MELODY_DATABASE.items():
        for track in tracks:
            if track['id'] != top_track['id']:
                # Calculate similarity score based on neural score, BPM, and key
                score_diff = abs(track['neural_score'] - top_track['neural_score'])
                bpm_diff = abs(track['bpm'] - top_track['bpm']) / 100
                
                similarity = max(0, 1 - (score_diff * 0.3 + bpm_diff * 0.7))
                
                if similarity > 0.6:  # Threshold for similarity
                    recommendations.append({
                        'track': track,
                        'category': category,
                        'similarity': similarity
                    })
    
    # Sort by similarity and return top 10
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    return recommendations[:10]

def neural_engagement_dashboard():
    """Display neural engagement analytics"""
    st.subheader("🧠 Neural Engagement Analytics")
    
    df = st.session_state.neural_data
    
    if df.empty:
        st.info("No listening data available yet. Start listening to tracks to see your neural engagement patterns!")
        return
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_engagement = df['neural_engagement'].mean()
        st.metric("Avg Engagement", f"{avg_engagement:.1f}/10")
    
    with col2:
        total_time = df['duration_listened'].sum() / 60
        st.metric("Total Listen Time", f"{total_time:.0f} min")
    
    with col3:
        best_category = df.groupby('category')['neural_engagement'].mean().idxmax()
        st.metric("Best Category", best_category)
    
    with col4:
        sessions = len(df)
        st.metric("Listening Sessions", sessions)
    
    # Engagement over time
    st.subheader("📈 Engagement Trends")
    daily_engagement = df.groupby(df['date'].dt.date)['neural_engagement'].mean().reset_index()
    
    fig_trend = px.line(daily_engagement, x='date', y='neural_engagement',
                       title='Daily Neural Engagement Trend',
                       labels={'neural_engagement': 'Neural Engagement Score'})
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Category comparison
    col1, col2 = st.columns(2)
    
    with col1:
        category_avg = df.groupby('category')['neural_engagement'].mean().reset_index()
        fig_cat = px.bar(category_avg, x='category', y='neural_engagement',
                        title='Average Engagement by Category',
                        color='neural_engagement',
                        color_continuous_scale='viridis')
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Top performing tracks
        top_tracks = df.groupby(['track_name', 'category'])['neural_engagement'].mean().reset_index()
        top_tracks = top_tracks.nlargest(10, 'neural_engagement')
        
        fig_top = px.bar(top_tracks, x='neural_engagement', y='track_name',
                        orientation='h', title='Top 10 Tracks by Engagement',
                        color='category')
        st.plotly_chart(fig_top, use_container_width=True)

def export_report():
    """Generate and export user report"""
    st.subheader("📊 Export Your Music Therapy Report")
    
    df = st.session_state.neural_data
    
    if df.empty:
        st.warning("No data available to export. Start listening to generate your report!")
        return
    
    # Generate report data
    report_data = {
        'user_summary': {
            'total_sessions': len(df),
            'total_listen_time_minutes': df['duration_listened'].sum() / 60,
            'average_engagement': df['neural_engagement'].mean(),
            'preferred_category': df.groupby('category')['neural_engagement'].mean().idxmax(),
            'most_played_track': df['track_name'].mode().iloc[0] if not df.empty else "N/A"
        },
        'category_performance': df.groupby('category').agg({
            'neural_engagement': ['mean', 'std', 'count'],
            'duration_listened': 'sum'
        }).round(2).to_dict(),
        'top_tracks': df.groupby(['track_name', 'category'])['neural_engagement'].mean().nlargest(10).to_dict(),
        'listening_patterns': df.groupby(df['date'].dt.hour)['neural_engagement'].mean().to_dict()
    }
    
    # Display report preview
    st.json(report_data)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download JSON Report"):
            st.download_button(
                label="Download Report",
                data=json.dumps(report_data, indent=2, default=str),
                file_name=f"music_therapy_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Download CSV Data"):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"neural_engagement_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

def general_user_dashboard():
    """Main dashboard for general users with music therapy features"""
    initialize_session_state()
    
    user_info = st.session_state.user_info
    
    st.title("🎵 Music Therapy Portal")
    st.markdown(f"Welcome, **{user_info['name']}**! Discover your optimal melodies for cognitive enhancement.")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🎼 Navigation")
        page = st.selectbox(
            "Select Feature",
            ["Dashboard", "Music Library", "My Playlists", "Neural Analytics", "Trend Analysis", "Export Report"]
        )
        
        st.markdown("---")
        st.markdown("### 🎛️ Quick Controls")
        
        # Current track display
        if st.session_state.current_track:
            st.markdown("**Now Playing:**")
            st.write(st.session_state.current_track['name'])
            
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.current_track = None
                st.session_state.is_playing = False
                st.session_state.playback_position = 0
        
        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    if page == "Dashboard":
        # Music player section
        if st.session_state.current_track:
            st.subheader("🎵 Now Playing")
            music_player_widget(st.session_state.current_track)
            st.markdown("---")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        df = st.session_state.neural_data
        
        with col1:
            sessions = len(df) if not df.empty else 0
            st.metric("Listening Sessions", sessions)
        
        with col2:
            avg_score = df['neural_engagement'].mean() if not df.empty else 0
            st.metric("Avg Neural Score", f"{avg_score:.1f}/10")
        
        with col3:
            total_time = df['duration_listened'].sum() / 60 if not df.empty else 0
            st.metric("Total Time", f"{total_time:.0f} min")
        
        with col4:
            categories = len(df['category'].unique()) if not df.empty else 0
            st.metric("Categories Explored", categories)
        
        st.markdown("---")
        
        # Featured recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌟 Recommended for You")
            
            # Show top tracks from user's preferred category
            if not df.empty:
                best_category = df.groupby('category')['neural_engagement'].mean().idxmax()
                recommended_tracks = MELODY_DATABASE[best_category][:3]
            else:
                recommended_tracks = MELODY_DATABASE["Classical"][:3]
            
            for track in recommended_tracks:
                track_card(track, best_category if not df.empty else "Classical")
        
        with col2:
            st.subheader("📊 Your Progress")
            
            if not df.empty:
                # Show engagement trend for last 7 days
                recent_data = df[df['date'] >= datetime.now() - timedelta(days=7)]
                if not recent_data.empty:
                    daily_avg = recent_data.groupby(recent_data['date'].dt.date)['neural_engagement'].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=daily_avg.index,
                        y=daily_avg.values,
                        mode='lines+markers',
                        name='Neural Engagement',
                        line=dict(color='#1f77b4', width=3)
                    ))
                    fig.update_layout(
                        title="7-Day Engagement Trend",
                        xaxis_title="Date",
                        yaxis_title="Neural Score",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Start listening to see your progress!")
            else:
                st.info("No data yet. Begin your music therapy journey!")
    
    elif page == "Music Library":
        st.subheader("🎼 Music Library")
        
        # Category filter
        selected_categories = st.multiselect(
            "Filter by Category",
            ["Classical", "Rock", "Pop", "Rap", "R&B"],
            default=["Classical", "Rock", "Pop", "Rap", "R&B"]
        )
        
        # Search
        search_term = st.text_input("🔍 Search tracks", placeholder="Enter track name...")
        
        # Display tracks by category
        for category in selected_categories:
            if category in MELODY_DATABASE:
                st.markdown(f"### {category} 🎵")
                
                tracks = MELODY_DATABASE[category]
                
                # Filter by search term
                if search_term:
                    tracks = [t for t in tracks if search_term.lower() in t['name'].lower()]
                
                if tracks:
                    for track in tracks:
                        track_card(track, category)
                else:
                    st.info(f"No tracks found matching '{search_term}' in {category} category.")
                
                st.markdown("---")
    
    elif page == "My Playlists":
        st.subheader("🎵 Smart Playlists")
        
        df = st.session_state.neural_data
        
        if not df.empty:
            # Find user's top track
            top_track_data = df.loc[df['neural_engagement'].idxmax()]
            top_track = None
            
            # Find the actual track object
            for category, tracks in MELODY_DATABASE.items():
                for track in tracks:
                    if track['id'] == top_track_data['track_id']:
                        top_track = track
                        break
            
            if top_track:
                st.markdown(f"### 🌟 Based on your top track: **{top_track['name']}**")
                st.markdown(f"Neural Engagement Score: **{top_track_data['neural_engagement']:.1f}/10**")
                
                recommendations = generate_playlist_recommendations(top_track)
                
                st.markdown("### 🎼 Recommended Playlist")
                
                for i, rec in enumerate(recommendations, 1):
                    col1, col2, col3 = st.columns([1, 6, 1])
                    
                    with col1:
                        st.write(f"**{i}.**")
                    
                    with col2:
                        track = rec['track']
                        st.markdown(f"""
                        **{track['name']}** ({rec['category']})  
                        Similarity: {rec['similarity']:.1%} | Neural Score: {track['neural_score']}/10
                        """)
                    
                    with col3:
                        if st.button("▶️", key=f"playlist_{track['id']}"):
                            st.session_state.current_track = track
                            st.session_state.is_playing = True
                            st.session_state.playback_position = 0
        else:
            st.info("Listen to some tracks first to generate personalized playlists!")
    
    elif page == "Neural Analytics":
        neural_engagement_dashboard()
    
    elif page == "Trend Analysis":
        st.subheader("📈 Trend Analysis")
        
        df = st.session_state.neural_data
        
        if not df.empty:
            # Time-based analysis
            st.markdown("### ⏰ Listening Patterns by Time of Day")
            
            hourly_engagement = df.groupby(df['date'].dt.hour)['neural_engagement'].mean()
            
            fig_hourly = px.line(
                x=hourly_engagement.index,
                y=hourly_engagement.values,
                title="Neural Engagement by Hour of Day",
                labels={'x': 'Hour of Day', 'y': 'Average Neural Engagement'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Weekly patterns
            st.markdown("### 📅 Weekly Patterns")
            
            df['day_of_week'] = df['date'].dt.day_name()
            weekly_engagement = df.groupby('day_of_week')['neural_engagement'].mean()
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_engagement = weekly_engagement.reindex([day for day in day_order if day in weekly_engagement.index])
            
            fig_weekly = px.bar(
                x=weekly_engagement.index,
                y=weekly_engagement.values,
                title="Average Engagement by Day of Week",
                labels={'x': 'Day of Week', 'y': 'Average Neural Engagement'}
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
            
            # Completion rate analysis
            st.markdown("### 🎯 Track Completion Analysis")
            
            completion_by_category = df.groupby('category')['completion_rate'].mean()
            
            fig_completion = px.pie(
                values=completion_by_category.values,
                names=completion_by_category.index,
                title="Average Completion Rate by Category"
            )
            st.plotly_chart(fig_completion, use_container_width=True)
        else:
            st.info("No data available for trend analysis. Start listening to generate insights!")
    
    elif page == "Export Report":
        export_report()
