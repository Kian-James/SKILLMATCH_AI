import '../config/firebase_config.dart';
import '../models/resume_model.dart';

class ResumeService {
  /// Upload resume text and get extracted skills
  static Future<ResumeModel> uploadResume(String userId, String resumeText) async {
    final data = {'resumeText': resumeText};

    final result = await FirebaseConfig.callFunction('processResume', data);

    return ResumeModel(
      userId: userId,
      skills: List<String>.from(result['skills'] ?? []),
      skillClusters: [], // optional
      lastUpdated: DateTime.now(),
    );
  }

  /// Fetch saved resume from Firestore
  static Future<ResumeModel?> getResume(String userId) async {
    final doc = await FirebaseConfig.firestore
        .collection('users')
        .doc(userId)
        .collection('resume')
        .doc('skills')
        .get();

    if (!doc.exists) return null;
    return ResumeModel.fromFirestore(doc);
  }
}
