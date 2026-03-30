// models/skill_gap.dart
// Data model for the skill gap analysis response from /skill-gap endpoint.

class SkillGapResult {
  final bool found;
  final String targetJob;
  final double completionScore;
  final List<String> matchedSkills;
  final List<String> missingSkills;
  final List<String> requiredSkills;
  final List<String> requiredCertifications;
  final int matchCount;
  final int totalRequired;
  final String message;

  const SkillGapResult({
    required this.found,
    required this.targetJob,
    required this.completionScore,
    required this.matchedSkills,
    required this.missingSkills,
    required this.requiredSkills,
    required this.requiredCertifications,
    required this.matchCount,
    required this.totalRequired,
    required this.message,
  });

  factory SkillGapResult.fromJson(Map<String, dynamic> json) {
    return SkillGapResult(
      found: json['found'] as bool? ?? false,
      targetJob: json['target_job'] as String? ?? '',
      completionScore: (json['completion_score'] as num?)?.toDouble() ?? 0.0,
      matchedSkills: List<String>.from(json['matched_skills'] ?? []),
      missingSkills: List<String>.from(json['missing_skills'] ?? []),
      requiredSkills: List<String>.from(json['required_skills'] ?? []),
      requiredCertifications: List<String>.from(json['required_certifications'] ?? []),
      matchCount: json['match_count'] as int? ?? 0,
      totalRequired: json['total_required'] as int? ?? 0,
      message: json['message'] as String? ?? '',
    );
  }

  /// How many skills are still missing
  int get missingCount => missingSkills.length;

  /// Completion as a 0.0–1.0 fraction for progress indicators
  double get completionFraction => (completionScore / 100).clamp(0.0, 1.0);
}

class LearningResource {
  final String title;
  final String type;
  final String platform;
  final String duration;
  final String description;

  const LearningResource({
    required this.title,
    required this.type,
    required this.platform,
    required this.duration,
    required this.description,
  });

  factory LearningResource.fromJson(Map<String, dynamic> json) {
    return LearningResource(
      title: json['title'] as String? ?? '',
      type: json['type'] as String? ?? 'Course',
      platform: json['platform'] as String? ?? '',
      duration: json['duration'] as String? ?? 'Self-paced',
      description: json['description'] as String? ?? '',
    );
  }

  String get emoji {
    switch (type.toLowerCase()) {
      case 'course':        return '🎓';
      case 'book':          return '📚';
      case 'video':         return '▶️';
      case 'documentation': return '📄';
      case 'tutorial':      return '🛠️';
      default:              return '🔗';
    }
  }
}