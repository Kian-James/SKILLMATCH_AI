lib/
├── main.dart                 # App entry point
├── app.dart                  # App widget, theme, routing
├── config/
│   └── firebase_config.dart  # Firebase initialization
├── models/
│   ├── resume_model.dart     # Resume skills & clusters
│   ├── job_model.dart        # Job posting data
│   └── skill_gap_model.dart  # Skill gap analysis data
├── services/
│   ├── auth_service.dart     # Firebase Authentication
│   ├── resume_service.dart   # Calls processResume cloud function
│   ├── job_service.dart      # Calls matchJobs function
│   └── skill_gap_service.dart# Calls analyzeSkillGap function
├── screens/
│   ├── login_screen.dart
│   ├── signup_screen.dart
│   ├── home_screen.dart
│   ├── resume_upload_screen.dart
│   ├── resume_result_screen.dart
│   ├── job_match_screen.dart
│   └── skill_gap_screen.dart
├── widgets/
│   ├── skill_chip.dart       # Displays a skill tag
│   ├── job_card.dart         # Job summary card
│   └── gap_card.dart         # Missing skill card
└── utils/
    ├── validators.dart       # Input validation
    └── helpers.dart          # Misc helpers (date formatting, etc.)
