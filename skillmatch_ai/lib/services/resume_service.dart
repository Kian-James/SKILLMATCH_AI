import '../config/firebase_config.dart';
import '../models/resume_model.dart';

class ResumeService {
  static const String processResumeUrl =
      'https://us-central1-yourproject.cloudfunctions.net/processResume';

  /// Upload resume text or Base64 file to backend
  static Future<ResumeModel> uploadResume(String userId, String resumeText) async {
    final data = {'resumeText': resumeText, 'userId': userId};

    final result = await FirebaseConfig.callFunction(processResumeUrl, data);

    return ResumeModel(
      userId: userId,
      skills: List<String>.from(result['skills'] ?? []),
      skillClusters: List<String>.from(result['skill_clusters'] ?? []),
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
