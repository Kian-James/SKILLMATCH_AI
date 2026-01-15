import 'package:cloud_firestore/cloud_firestore.dart';

class SkillGapModel {
  final String role;
  final int totalJobs;
  final List<String> presentSkills;
  final List<String> missingSkills;
  final List<String> suggestedCertifications;
  final DateTime analyzedAt;

  SkillGapModel({
    required this.role,
    required this.totalJobs,
    required this.presentSkills,
    required this.missingSkills,
    required this.suggestedCertifications,
    required this.analyzedAt,
  });

  factory SkillGapModel.fromFirestore(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>;
    return SkillGapModel(
      role: data['role'] ?? '',
      totalJobs: data['totalJobs'] ?? 0,
      presentSkills: List<String>.from(data['presentSkills'] ?? []),
      missingSkills: List<String>.from(data['missingSkills'] ?? []),
      suggestedCertifications: List<String>.from(data['suggestedCertifications'] ?? []),
      analyzedAt: (data['analyzed_at'] as Timestamp?)?.toDate() ?? DateTime.now(),
    );
  }

  Map<String, dynamic> toFirestore() {
    return {
      'role': role,
      'totalJobs': totalJobs,
      'presentSkills': presentSkills,
      'missingSkills': missingSkills,
      'suggestedCertifications': suggestedCertifications,
      'analyzed_at': analyzedAt,
    };
  }
}
