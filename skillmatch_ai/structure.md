lib/
│
├── main.dart
│
├── app/
│   ├── app.dart
│   └── routes.dart
│
├── config/
│   ├── api_config.dart
│   ├── app_constants.dart
│   └── algorithm_modes.dart
│
├── core/
│   ├── services/
│   │   ├── api_service.dart
│   │   ├── ocr_service.dart
│   │   ├── document_scan_service.dart
│   │   └── local_storage_service.dart
│   │
│   ├── utils/
│   │   ├── text_normalizer.dart
│   │   ├── skill_extractor.dart
│   │   ├── certification_extractor.dart
│   │   ├── timer_util.dart
│   │   └── file_helper.dart
│   │
│   └── models/
│       ├── skill_model.dart
│       ├── certification_model.dart
│       ├── job_model.dart
│       ├── match_result_model.dart
│       └── benchmark_model.dart
│
├── features/
│   ├── resume_input/
│   │   ├── view/
│   │   │   ├── resume_upload_screen.dart
│   │   │   ├── resume_scan_screen.dart
│   │   │   └── resume_preview_screen.dart
│   │   │
│   │   ├── viewmodel/
│   │   │   └── resume_viewmodel.dart
│   │   │
│   │   └── resume_module.dart
│   │
│   ├── job_matching/
│   │   ├── view/
│   │   │   ├── job_result_screen.dart
│   │   │   └── job_detail_screen.dart
│   │   │
│   │   ├── viewmodel/
│   │   │   └── job_match_viewmodel.dart
│   │   │
│   │   └── job_matching_module.dart
│   │
│   ├── skill_gap/
│   │   ├── view/
│   │   │   └── skill_gap_screen.dart
│   │   │
│   │   ├── viewmodel/
│   │   │   └── skill_gap_viewmodel.dart
│   │   │
│   │   └── skill_gap_module.dart
│   │
│   ├── benchmark/
│   │   ├── view/
│   │   │   └── benchmark_screen.dart
│   │   │
│   │   ├── viewmodel/
│   │   │   └── benchmark_viewmodel.dart
│   │   │
│   │   └── benchmark_module.dart
│
├── shared/
│   ├── widgets/
│   │   ├── primary_button.dart
│   │   ├── skill_chip.dart
│   │   ├── certification_chip.dart
│   │   ├── similarity_bar.dart
│   │   └── loading_indicator.dart
│   │
│   └── theme/
│       ├── app_colors.dart
│       └── app_theme.dart
│
└── storage/
├── local_skill_store.dart
├── local_certification_store.dart
└── local_benchmark_store.dart
