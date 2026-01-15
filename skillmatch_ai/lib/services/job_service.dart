import '../config/firebase_config.dart';
import '../models/job_model.dart';

class JobService {
  static const String matchJobsUrl =
      'https://us-central1-yourproject.cloudfunctions.net/matchJobs';

  /// Call backend to get jobs matched to user
  static Future<List<JobModel>> getMatchedJobs(String userId) async {
    final result = await FirebaseConfig.callFunction(matchJobsUrl, {'userId': userId});

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
