// models/growth_data.dart
// Data model for growth tracking response from /growth-tracking endpoint.

class GrowthData {
  final bool hasData;
  final String message;
  final int totalSnapshots;
  final String? firstAnalysis;
  final String? latestAnalysis;
  final int currentSkillCount;
  final double currentBestMatchScore;
  final int skillCountDelta;
  final double scoreDelta;
  final List<String> newSkillsAdded;
  final int newSkillsCount;
  final int improvementStreak;
  final List<GrowthSnapshot> timeSeries;

  const GrowthData({
    required this.hasData,
    required this.message,
    required this.totalSnapshots,
    this.firstAnalysis,
    this.latestAnalysis,
    required this.currentSkillCount,
    required this.currentBestMatchScore,
    required this.skillCountDelta,
    required this.scoreDelta,
    required this.newSkillsAdded,
    required this.newSkillsCount,
    required this.improvementStreak,
    required this.timeSeries,
  });

  factory GrowthData.fromJson(Map<String, dynamic> json) {
    return GrowthData(
      hasData: json['has_data'] as bool? ?? false,
      message: json['message'] as String? ?? '',
      totalSnapshots: json['total_snapshots'] as int? ?? 0,
      firstAnalysis: json['first_analysis'] as String?,
      latestAnalysis: json['latest_analysis'] as String?,
      currentSkillCount: json['current_skill_count'] as int? ?? 0,
      currentBestMatchScore:
      (json['current_best_match_score'] as num?)?.toDouble() ?? 0.0,
      skillCountDelta: json['skill_count_delta'] as int? ?? 0,
      scoreDelta: (json['score_delta'] as num?)?.toDouble() ?? 0.0,
      newSkillsAdded: List<String>.from(json['new_skills_added'] ?? []),
      newSkillsCount: json['new_skills_count'] as int? ?? 0,
      improvementStreak: json['improvement_streak'] as int? ?? 0,
      timeSeries: (json['time_series'] as List<dynamic>? ?? [])
          .map((s) => GrowthSnapshot.fromJson(s as Map<String, dynamic>))
          .toList(),
    );
  }

  /// Best match score as percentage string
  String get matchScorePercent =>
      '${(currentBestMatchScore * 100).toStringAsFixed(0)}%';

  /// Score delta as percentage string with sign
  String get scoreDeltaPercent {
    final pct = (scoreDelta * 100).round();
    return pct >= 0 ? '+$pct%' : '$pct%';
  }
}

class GrowthSnapshot {
  final String timestamp;
  final int skillCount;
  final double bestMatchScore;
  final int certificationsCount;
  final List<String> skillsSample;

  const GrowthSnapshot({
    required this.timestamp,
    required this.skillCount,
    required this.bestMatchScore,
    required this.certificationsCount,
    required this.skillsSample,
  });

  factory GrowthSnapshot.fromJson(Map<String, dynamic> json) {
    return GrowthSnapshot(
      timestamp: json['timestamp'] as String? ?? '',
      skillCount: json['skill_count'] as int? ?? 0,
      bestMatchScore: (json['best_match_score'] as num?)?.toDouble() ?? 0.0,
      certificationsCount: json['certifications_count'] as int? ?? 0,
      skillsSample: List<String>.from(json['skills_sample'] ?? []),
    );
  }

  /// Score as a 0–100 value for charting
  double get scorePercent => bestMatchScore * 100;
}