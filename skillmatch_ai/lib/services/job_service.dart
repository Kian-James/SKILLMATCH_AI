import '../config/firebase_config.dart';
import '../models/job_model.dart';

class JobService {
  /// Call matchJobs cloud function for a user
  static Future<List<JobModel>> getMatchedJobs(String userId) async {
    final result = await FirebaseConfig.callFunction('matchJobs', {'userId': userId});

    final jobs = <JobModel>[];
    for (var job in result) {
      jobs.add(JobModel(
        jobId: job['job_id'],
        title: job['title'],
        company: job['company'],
        location: job['location'],
        skills: List<String>.from(job['overlapping_skills'] ?? []),
      ));
    }
    return jobs;
  }
}
