import 'package:flutter/material.dart';
import '../services/job_service.dart';
import '../models/job_model.dart';
import '../widgets/job_card.dart';

class JobMatchScreen extends StatefulWidget {
  final String userId;

  const JobMatchScreen({Key? key, required this.userId}) : super(key: key);

  @override
  _JobMatchScreenState createState() => _JobMatchScreenState();
}

class _JobMatchScreenState extends State<JobMatchScreen> {
  bool _isLoading = true;
  List<JobModel> _jobs = [];
  String _error = '';

  @override
  void initState() {
    super.initState();
    _fetchJobs();
  }

  Future<void> _fetchJobs() async {
    setState(() {
      _isLoading = true;
      _error = '';
    });

    try {
      final jobs = await JobService.getMatchedJobs(widget.userId);
      setState(() {
        _jobs = jobs;
      });
    } catch (e) {
      setState(() {
        _error = 'Error fetching jobs: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Job Matches')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: _isLoading
            ? const Center(child: CircularProgressIndicator())
            : _error.isNotEmpty
            ? Center(child: Text(_error))
            : _jobs.isEmpty
            ? const Center(child: Text('No matching jobs found'))
            : ListView.builder(
          itemCount: _jobs.length,
          itemBuilder: (context, index) {
            final job = _jobs[index];
            return Padding(
              padding: const EdgeInsets.only(bottom: 12.0),
              child: JobCard(job: job),
            );
          },
        ),
      ),
    );
  }
}
