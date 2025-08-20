# Healthcare Portal - Music Therapy App with Google Sign-In

A Streamlit web application with Google OAuth authentication that routes users to different dashboards based on their email addresses. Features an advanced music therapy system for general users with neural engagement tracking.

## Features

- **Google OAuth Authentication**: Secure login with Google accounts
- **Role-based Access Control**: Different interfaces for caregivers and general users
- **Caregiver Dashboard**: Patient management, appointments, reports, and settings
- **Music Therapy Portal**: Advanced music therapy system for general users
- **Email-based Routing**: Automatic redirection based on user email domain

### Music Therapy Features (General Users)
- **Melody Database**: 45+ curated melodies across 5 music genres (Classical, Rock, Pop, Rap, R&B)
- **Interactive Music Player**: Play/pause controls, progress bars, volume sliders
- **Neural Engagement Tracking**: Monitor brain response to different melodies
- **Smart Playlist Generation**: AI-powered recommendations based on highest engagement scores
- **Visual Analytics**: Charts and graphs showing listening patterns and neural engagement
- **Trend Analysis**: Time-based patterns, weekly trends, and completion rates
- **Export Reports**: Download detailed analytics in JSON or CSV format
- **Genre-based Organization**: Browse music by therapeutic music genres

### ML-Powered Caregiver Features
- **Pre-trained Random Forest Model**: Uses default trained model for music genre predictions (Classical, Rock, Pop, Rap, R&B)
- **Patient-Specific EEG Data Management**: Upload and manage EEG data for individual patients
- **EEG Data Processing**: Automated analysis of brain wave frequency bands (Delta, Theta, Alpha, Beta, Gamma)
- **Cognitive Score Calculations**:
  - **Engagement Score**: `Mean(Beta+Gamma) - Mean(Alpha+Theta)` - measures mental alertness
  - **Focus Score**: `Theta/Beta Ratio` - measures sustained attention and concentration
  - **Relaxation Score**: `Mean(Alpha+Theta) - Mean(Beta+Gamma)` - measures calmness and passive mental states
- **Multi-Patient Support**: Manage data for multiple patients with individual tracking
- **Patient Selection Interface**: Choose existing patients or create new patient profiles
- **ML Model Performance Dashboard**: View model accuracy, capabilities, and performance metrics
- **Cognitive Insights Visualization**: Interactive charts showing brain response patterns per patient
- **Real EEG Data Upload**: CSV import functionality for authentic EEG datasets (no synthetic data)
- **Patient Summary Dashboard**: Overview of all patients with average cognitive scores
- **Comprehensive Reporting**: Export detailed patient analysis and ML predictions

## Required Files

### Pre-trained ML Model
Place a pre-trained Random Forest model file named `random_forest_model.pkl` in the project root directory. This model should be trained to predict music genres (1=Classical, 2=Rock, 3=Pop, 4=Rap, 5=R&B) from EEG frequency band features.

### EEG Data Format
For caregiver uploads, CSV files should contain:
- **40 EEG columns**: Frequency bands (Delta, Theta, Alpha, Beta, Gamma) for 4 electrodes (TP9, AF7, AF8, TP10)
- **Format**: `{Band}_{Electrode}_mean` (e.g., `Delta_TP9_mean`, `Alpha_AF7_mean`)
- **Target column**: `Melody` (1-5 for music genres)
- **No metadata**: Patient ID assigned automatically during upload

## Demo Mode

For testing purposes, the app includes a demo login form that doesn't require actual Google OAuth setup. Simply enter any email and name to simulate login.

## User Roles

### Caregivers
- **ML-Powered Analytics**: Access to Random Forest model for music therapy predictions
- **Patient-Specific EEG Management**: Upload and analyze EEG data for individual patients
- **Cognitive Assessment Tools**: Calculate engagement, focus, and relaxation scores
- **Multi-Patient Dashboard**: Manage multiple patients with individual profiles
- **Real Data Analysis**: Process authentic EEG datasets without synthetic data generation
- **Performance Monitoring**: View ML model accuracy and prediction capabilities
- **Export Functionality**: Generate detailed reports for clinical use

### General Users (Patients)
- **Music Therapy Portal**: Access to 5-genre melody database (Classical, Rock, Pop, Rap, R&B)
- **Interactive Music Player**: Full-featured audio player with controls
- **Neural Engagement Tracking**: Monitor personal brain response to music
- **Smart Recommendations**: AI-generated playlists based on engagement patterns
- **Personal Analytics**: View listening trends and neural engagement data
- **Progress Tracking**: Monitor therapeutic progress over time
