// models/resume_analysis.dart
// Data model for the full resume analysis response from /analyze endpoint.

class ResumeAnalysis {
  final String userId;
  final List<String> skills;
  final List<String> certifications;
  final List<JobMatch> topJobMatches;
  final double bestMatchScore;
  final String snapshotId;
  final String message;

  const ResumeAnalysis({
    required this.userId,
    required this.skills,
    required this.certifications,
    required this.topJobMatches,
    required this.bestMatchScore,
    required this.snapshotId,
    required this.message,
  });

  factory ResumeAnalysis.fromJson(Map<String, dynamic> json) {
    return ResumeAnalysis(
      userId: json['user_id'] as String? ?? '',
      skills: List<String>.from(json['skills'] ?? []),
      certifications: List<String>.from(json['certifications'] ?? []),
      topJobMatches: (json['top_job_matches'] as List<dynamic>? ?? [])
          .map((j) => JobMatch.fromJson(j as Map<String, dynamic>))
          .toList(),
      bestMatchScore: (json['best_match_score'] as num?)?.toDouble() ?? 0.0,
      snapshotId: json['snapshot_id'] as String? ?? '',
      message: json['message'] as String? ?? '',
    );
  }
}

class JobMatch {
  final String jobTitle;
  final double matchScore;
  final List<String> skills;
  final List<String> certifications;

  const JobMatch({
    required this.jobTitle,
    required this.matchScore,
    required this.skills,
    required this.certifications,
  });

  factory JobMatch.fromJson(Map<String, dynamic> json) {
    return JobMatch(
      jobTitle: json['job_title'] as String? ?? '',
      matchScore: (json['match_score'] as num?)?.toDouble() ?? 0.0,
      skills: List<String>.from(json['skills'] ?? []),
      certifications: List<String>.from(json['certifications'] ?? []),
    );
  }

  Map<String, dynamic> toJson() => {
    'job_title': jobTitle,
    'match_score': matchScore,
    'skills': skills,
    'certifications': certifications,
  };

  /// Score category for color coding
  ScoreLevel get scoreLevel {
    if (matchScore >= 80) return ScoreLevel.high;
    if (matchScore >= 60) return ScoreLevel.medium;
    return ScoreLevel.low;
  }
}

enum ScoreLevel { high, medium, low }