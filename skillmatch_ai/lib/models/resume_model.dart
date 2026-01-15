import 'package:cloud_firestore/cloud_firestore.dart';

class ResumeModel {
  final String userId;
  final List<String> skills;
  final List<String> skillClusters;
  final DateTime lastUpdated;

  ResumeModel({
    required this.userId,
    required this.skills,
    required this.skillClusters,
    required this.lastUpdated,
  });

  factory ResumeModel.fromFirestore(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>;
    return ResumeModel(
      userId: data['user_id'] ?? '',
      skills: List<String>.from(data['skills'] ?? []),
      skillClusters: List<String>.from(data['skill_clusters'] ?? []),
      lastUpdated: (data['last_updated'] as Timestamp?)?.toDate() ?? DateTime.now(),
    );
  }

  Map<String, dynamic> toFirestore() {
    return {
      'user_id': userId,
      'skills': skills,
      'skill_clusters': skillClusters,
      'last_updated': lastUpdated,
    };
  }
}
