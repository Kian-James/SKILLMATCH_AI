import '../config/firebase_config.dart';
import '../models/skill_gap_model.dart';

class SkillGapService {
  static const String analyzeSkillGapUrl =
      'https://us-central1-yourproject.cloudfunctions.net/analyzeSkillGap';

  /// Analyze skill gap for a user and specific job role
  static Future<SkillGapModel> analyzeSkillGap(String userId, String role) async {
    final result = await FirebaseConfig.callFunction(analyzeSkillGapUrl, {
      'userId': userId,
      'role': role,
    });

    return SkillGapModel(
      role: result['role'] ?? '',
      totalJobs: result['totalJobs'] ?? 0,
      presentSkills: List<String>.from(result['presentSkills'] ?? []),
      missingSkills: List<String>.from(result['missingSkills'] ?? []),
      suggestedCertifications: List<String>.from(result['suggestedCertifications'] ?? []),
      analyzedAt: DateTime.parse(result['analyzed_at'] ?? DateTime.now().toIso8601String()),
    );
  }
}
