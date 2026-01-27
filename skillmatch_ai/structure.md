frontend/
├── lib/
│   ├── main.dart
│   │
│   ├── app/
│   │   ├── app.dart              # Root widget
│   │   ├── routes.dart           # App navigation
│   │   └── theme.dart
│   │
│   ├── core/
│   │   ├── config.dart           # API URLs & flags
│   │   ├── constants.dart
│   │   └── privacy_notice.dart   # Privacy disclaimer
│   │
│   ├── data/
│   │   ├── models/
│   │   │   ├── job.dart
│   │   │   ├── skill_gap.dart
│   │   │   └── learning_path.dart
│   │   │
│   │   ├── services/
│   │   │   ├── api_service.dart  # FastAPI client
│   │   │   └── upload_service.dart
│   │
│   ├── features/
│   │   ├── resume_upload/
│   │   │   ├── upload_screen.dart
│   │   │   ├── upload_controller.dart
│   │   │   └── upload_state.dart
│   │   │
│   │   ├── job_matching/
│   │   │   ├── job_list_screen.dart
│   │   │   ├── job_card.dart
│   │   │   └── match_score_badge.dart
│   │   │
│   │   ├── career_gap/
│   │   │   ├── gap_screen.dart
│   │   │   ├── missing_skills.dart
│   │   │   ├── certifications.dart
│   │   │   └── learning_path_view.dart
│   │   │
│   │   └── settings/
│   │       ├── algorithm_toggle.dart   # SBERT / KNN switch
│   │       └── performance_info.dart
│   │
│   ├── shared/
│   │   ├── widgets/
│   │   │   ├── loading.dart
│   │   │   ├── error.dart
│   │   │   └── section_header.dart
│   │   │
│   │   └── utils/
│   │       ├── file_picker.dart
│   │       └── formatters.dart
│   │
│   └── state/
│       ├── app_state.dart
│       └── algorithm_state.dart
│
├── pubspec.yaml
└── README.md
