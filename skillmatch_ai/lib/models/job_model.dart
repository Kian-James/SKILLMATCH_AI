import 'package:cloud_firestore/cloud_firestore.dart';

class JobModel {
  final String jobId;
  final String title;
  final String company;
  final String location;
  final List<String> skills;

  JobModel({
    required this.jobId,
    required this.title,
    required this.company,
    required this.location,
    required this.skills,
  });

  factory JobModel.fromFirestore(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>;
    return JobModel(
      jobId: doc.id,
      title: data['title'] ?? '',
      company: data['company'] ?? '',
      location: data['location'] ?? '',
      skills: List<String>.from(data['skills'] ?? []),
    );
  }

  Map<String, dynamic> toFirestore() {
    return {
      'title': title,
      'company': company,
      'location': location,
      'skills': skills,
    };
  }
}
