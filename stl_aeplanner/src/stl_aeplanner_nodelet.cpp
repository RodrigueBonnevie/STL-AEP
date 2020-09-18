#include <stl_aeplanner/stl_aeplanner_nodelet.h>

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(stl_aeplanner::STLAEPlannerNodelet, nodelet::Nodelet)

namespace stl_aeplanner
{
void STLAEPlannerNodelet::onInit()
{
  NODELET_DEBUG("Initializing nodelet STLAEPlanner...");

  ros::NodeHandle& nh = getMTNodeHandle();  // getNodeHandle();
  stl_aeplanner_ = new STLAEPlanner(nh);
}
STLAEPlannerNodelet::~STLAEPlannerNodelet()
{
  std::string path = "/home/rodrigue/catkin_ws/src/thesis/ufomaps/";
  stl_aeplanner_->writeMap(path);
  delete stl_aeplanner_;
}
}  // namespace stl_aeplanner
