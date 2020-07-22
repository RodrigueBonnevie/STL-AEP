#include <stl_aeplanner/stl_aeplanner.h> 
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <stl_aeplanner_msgs/LTLStats.h>
#include <numeric>

#include <queue>
#include <algorithm>

#include <omp.h>


namespace stl_aeplanner
{
STLAEPlanner::STLAEPlanner(const ros::NodeHandle& nh)
  : nh_(nh)
  , as_(nh_, "make_plan", boost::bind(&STLAEPlanner::execute, this, _1), false)
  //, octomap_sub_(nh_.subscribe("octomap", 1, &STLAEPlanner::octomapCallback, this))
  , agent_pose_sub_(nh_.subscribe("agent_pose", 1, &STLAEPlanner::agentPoseCallback, this))
  , router_sub_(nh_.subscribe("/router", 10, &STLAEPlanner::routerCallback, this))
  , transform_sub_(nh_.subscribe("/transform", 1, &STLAEPlanner::transformCallback, this))
  , ufomap_sub_(nh_.subscribe("/virtual_camera/depth/points", 10, &STLAEPlanner::ufomapCallback, this))
  , rrt_marker_pub_(nh_.advertise<visualization_msgs::MarkerArray>("rrtree", 1000))
  , gain_pub_(nh_.advertise<stl_aeplanner_msgs::Node>("gain_node", 1000))
  , gp_query_client_(nh_.serviceClient<stl_aeplanner_msgs::Query>("gp_query_server"))
  , reevaluate_server_(nh_.advertiseService("reevaluate", &STLAEPlanner::reevaluate, this))
  , best_node_client_(nh_.serviceClient<stl_aeplanner_msgs::BestNode>("best_node_server"))
  , current_state_initialized_(false)
//  , ot_(NULL)
  , ltl_cs_(nh_)
  , ltl_path_pub_(nh_.advertise<nav_msgs::Path>("ltl_path", 1000))
  , ltl_stats_pub_(nh_.advertise<stl_aeplanner_msgs::LTLStats>("ltl_stats", 1000))
  , ltl_iterations_(0)
  , ufomap_(0.2)
  , ufomap_viz_(nh_)
  , session_manager_(nh_)
{
  tfl_ = new tf2_ros::TransformListener(tfBuffer_);
  // Set up dynamic reconfigure server
  ltl_f_ = boost::bind(&STLAEPlanner::configCallback, this, _1, _2);
  ltl_cs_.setCallback(ltl_f_);

  params_ = readParams();
  max_sampling_radius_squared_ = pow(params_.max_sampling_radius, 2.0);

  home_pose_v4[0] = 0; home_pose_v4[1] = 0; 
  home_pose_v4[2] = 1; home_pose_v4[4] = 0; 
  as_.start();
}

void STLAEPlanner::execute(const stl_aeplanner_msgs::aeplannerGoalConstPtr& goal)
{
  //std::shared_ptr<octomap::OcTree> ot = ot_;
  Eigen::Vector4d current_state = current_state_;

  ROS_ERROR_STREAM("Execute start!");
  stl_aeplanner_msgs::aeplannerResult result;

  // Check if aeplanner has recieved agent's pose yet
  if (!current_state_initialized_)
  {
    ROS_WARN("Agent's pose not yet received");
    ROS_WARN("Make sure it is being published and correctly mapped");
    as_.setSucceeded(result);
    return;
  }
  if (0 == ufomap_.getNumLeafNodes())
  {
    ROS_WARN("No octomap received");
    as_.setSucceeded(result);
    return;
  }

  ufomap::Point3d min(current_state[0] - params_.max_sampling_radius - ltl_max_search_distance_,
                       current_state[1] - params_.max_sampling_radius - ltl_max_search_distance_,
                       current_state[2] - params_.max_sampling_radius - ltl_max_search_distance_);

  ufomap::Point3d max(current_state[0] + params_.max_sampling_radius + ltl_max_search_distance_,
                       current_state[1] + params_.max_sampling_radius + ltl_max_search_distance_,
                       current_state[2] + params_.max_sampling_radius + ltl_max_search_distance_);

  std::shared_ptr<point_rtree> stl_rtree = std::make_shared<point_rtree>(getRtreeUfomap(min, max));

  value_rtree rtree;

  ROS_WARN("Init");
  root_ = initialize(&rtree, stl_rtree, current_state);
  ROS_WARN("expandRRT");
  if (!goal->session_done){ // explore
      expandRRTUfomap(&rtree, stl_rtree, current_state);
  }
  if (goal->session_done){ // return home
      Eigen::Vector4d dist_to_home_4 = current_state - home_pose_v4;
      Eigen::Vector3d dist_to_home_3(dist_to_home_4[0],
                                      dist_to_home_4[1],
                                      dist_to_home_4[2]);
      if (dist_to_home_3.norm() < 1.3*params_.extension_range){ // Update dynamic parameters here
          ROS_INFO("Returned back home, updating dynamic parameters");
          ++ session_number_;
          ufomap_mutex_.lock();
          ufomap_.update_dynamic_parameters(session_number_);
          evaluate::evaluate_dynamic_map(&ufomap_, session_manager_.dynamic_models_);
          ufomap_mutex_.unlock();
          session_manager_.refurnish();
          going_home_ = false;
          result.at_home = true;
          ros::Duration(2).sleep();
          best_node_ = nullptr;
          expandRRTUfomap(&rtree, stl_rtree, current_state);
      }
      else
      {
          going_home_ = true;
          ROS_INFO("stl_aeplanner, going back home");
          planToGoalRRT(&rtree, stl_rtree, current_state);
      }
  }

  ROS_WARN("getCopyOfParent");
  if (!best_node_){
    ROS_ERROR("Best node is nullptr");
  }
  best_branch_root_ = best_node_->getCopyOfParentBranch();

  ROS_WARN("createRRTMarker");
  rrt_marker_pub_.publish(
      createRRTMarkerArray(root_, stl_rtree, current_state, ltl_lambda_, ltl_min_distance_, ltl_max_distance_,
                           ltl_min_distance_active_, ltl_max_distance_active_, ltl_max_search_distance_,
                           params_.bounding_radius, ltl_step_size_, ltl_routers_, ltl_routers_active_, params_.lambda,
                           ltl_min_altitude_active_, ltl_max_altitude_active_, ltl_min_altitude_, ltl_max_altitude_));
  ROS_WARN("publishRecursive");
  publishEvaluatedNodesRecursive(root_);

  ROS_WARN("extractPose");
  result.pose.pose = vecToPose(best_branch_root_->children_[0]->state_);
  if (best_node_->score(stl_rtree, ltl_lambda_, ltl_min_distance_, ltl_max_distance_, ltl_min_distance_active_,
                        ltl_max_distance_active_, ltl_max_search_distance_, params_.bounding_radius, ltl_step_size_,
                        ltl_routers_, ltl_routers_active_, params_.lambda, ltl_min_altitude_, ltl_max_altitude_,
                        ltl_min_altitude_active_, ltl_max_altitude_active_) > params_.zero_gain)
    result.is_clear = true;
  else
  {
    ROS_WARN("Getting frontiers");
    result.frontiers = getFrontiers();
    result.is_clear = false;
    best_branch_root_ = NULL;
  }
  as_.setSucceeded(result);
}

//point_rtree STLAEPlanner::getRtree(std::shared_ptr<octomap::OcTree> ot, octomap::point3d min, octomap::point3d max)
//{
//  point_rtree rtree;
//  for (octomap::OcTree::leaf_bbx_iterator it = ot->begin_leafs_bbx(min, max), it_end = ot->end_leafs_bbx();
//       it != it_end; ++it)
//  {
//    if (it->getLogOdds() > 0)
//    {
//      rtree.insert(point(it.getX(), it.getY(), it.getZ()));
//    }
//  }
//
//  return rtree;
//}

point_rtree STLAEPlanner::getRtreeUfomap(ufomap::Point3f min, ufomap::Point3f max)
{
  point_rtree rtree;
  ufomap_mutex_.lock();
  for(auto it = ufomap_.begin_leafs_bbx(min, max); it != ufomap_.end_leafs_bbx(); ++it) 
  {
    //if (it.isFree()) // TODO this is right right?
    if (it.isOccupied()) // TODO this is right right?
    {
      rtree.insert(point(it.getX(), it.getY(), it.getZ()));
    }
  }
  ufomap_mutex_.unlock();

  return rtree;
}

std::shared_ptr<RRTNode> STLAEPlanner::initialize(value_rtree* rtree, 
                                                    std::shared_ptr<point_rtree> stl_rtree,
                                                    const Eigen::Vector4d& current_state)
{
  best_node_ = nullptr;

  std::shared_ptr<RRTNode> root;
  if (best_branch_root_ and !best_branch_root_->children_.empty())
  {
    // Initialize with previous best branch

    // Discard root node from tree since we just were there...
    
    root = best_branch_root_->children_[0];
    root->parent_ = NULL;
    best_branch_root_->children_.clear();
    if (going_home_)
    {
        initializeKDTreeWithPreviousBestBranch(rtree,stl_rtree, root);
        return root;
    }

    initializeKDTreeWithPreviousBestBranch(rtree,stl_rtree, root);
    reevaluatePotentialInformationGainRecursive(root);
  }
  else
  {
    // Initialize without any previous branch
    root = std::make_shared<RRTNode>();
    root->state_[0] = current_state[0];
    root->state_[1] = current_state[1];
    root->state_[2] = current_state[2];
    rtree->insert(std::make_pair(point(root->state_[0], root->state_[1], root->state_[2]), root));
  }
  return root;
}

void STLAEPlanner::initializeKDTreeWithPreviousBestBranch(value_rtree* rtree, 
                                                            std::shared_ptr<point_rtree> stl_rtree, 
                                                            std::shared_ptr<RRTNode> root)
{
  std::shared_ptr<RRTNode> current_node = root;
  std::shared_ptr<RRTNode> next_node;
  do
  {
    rtree->insert(
        std::make_pair(point(current_node->state_[0], current_node->state_[1], current_node->state_[2]), current_node));

    if (!current_node->children_.empty()){
        next_node = current_node->children_[0];
        if (collisionLine(stl_rtree, current_node->state_, next_node->state_, params_.bounding_radius)){
            //ROS_INFO("stl_aeplanner, break best branch");
            break;
        }
        current_node = next_node;
    }

  } while (!current_node->children_.empty());
}

void STLAEPlanner::reevaluatePotentialInformationGainRecursive(std::shared_ptr<RRTNode> node)
{
  std::pair<double, double> ret = gainCubatureUfomap(node->state_);
  node->state_[3] = ret.second;  // Assign yaw angle that maximizes g
  node->gain_ = ret.first;
  for (typename std::vector<std::shared_ptr<RRTNode>>::iterator node_it = node->children_.begin();
       node_it != node->children_.end(); ++node_it)
    reevaluatePotentialInformationGainRecursive(*node_it);
}

void STLAEPlanner::planToGoalRRT(value_rtree* rtree,
                                 std::shared_ptr<point_rtree> stl_rtree,
                                 const Eigen::Vector4d& current_state)
{
  // Expand an RRT tree and calculate information gain in every node
  int max_it = 3000;
  for (int n = 0;
       (n < max_it) and ros::ok();
       ++n)
  {
    std::shared_ptr<RRTNode> new_node = std::make_shared<RRTNode>();
    std::shared_ptr<RRTNode> nearest;
    const ufomap::OccupancyNode* ot_result;

    // Sample new point around agent and check that
    // Don't need this check here, (1) it is within the boundaries
    // (2) it is in known space
    // (3) the path between the new node and it')s parent does not contain any
    // obstacles
    do
    {
      Eigen::Vector4d offset;
          if (rand()%100 < 20)
          {
              offset = sampleNewPoint();
          }
          else
          {
              offset = home_pose_v4 - current_state;
          }
      new_node->state_ = current_state + offset;
      nearest = chooseParent(*rtree, stl_rtree, new_node, params_.extension_range);
      new_node->state_ = restrictDistance(nearest->state_, new_node->state_);
      //ufomap_mutex_.lock();
      ot_result = ufomap_.search(ufomap::Point3f(new_node->state_[0], new_node->state_[1], new_node->state_[2]));
      //ufomap_mutex_.unlock();
      if (ot_result == NULL){
        continue;
      }
    } while (!isInsideBoundaries(new_node->state_) or (!ot_result or !ufomap_.isFree(*ot_result)) or
             collisionLine(stl_rtree, nearest->state_, new_node->state_, params_.bounding_radius));

    // new_node is now ready to be added to tree
    new_node->parent_ = nearest;
    nearest->children_.push_back(new_node);

    // rewire tree with new node
    rewire(*rtree, stl_rtree, new_node, params_.extension_range, params_.bounding_radius, params_.d_overshoot_);

    // Calculate potential information gain for new_node
    std::pair<double, double> ret = getGain(new_node);
    new_node->state_[3] = ret.second;  // Assign yaw angle that maximizes g
    new_node->gain_ = ret.first;
    rtree->insert(std::make_pair(point(new_node->state_[0], new_node->state_[1], new_node->state_[2]), new_node));

    // Update best node
    
      Eigen::Vector3d home_pose_v3(home_pose_v4[0],
                                        home_pose_v4[1],
                                        home_pose_v4[2]);
      Eigen::Vector3d state_v3(new_node->state_[0],
                                new_node->state_[1],
                                new_node->state_[2]);
      Eigen::Vector3d dist_to_home = home_pose_v3 - state_v3;
      // Update best node
      if (!best_node_ or
          new_node->score(stl_rtree, ltl_lambda_, ltl_min_distance_, ltl_max_distance_, ltl_min_distance_active_,
                          ltl_max_distance_active_, ltl_max_search_distance_, params_.bounding_radius, ltl_step_size_,
                          ltl_routers_, ltl_routers_active_, params_.lambda, ltl_min_altitude_, ltl_max_altitude_,
                          ltl_min_altitude_active_, ltl_max_altitude_active_) >
              best_node_->score(stl_rtree, ltl_lambda_, ltl_min_distance_, ltl_max_distance_, ltl_min_distance_active_,
                                ltl_max_distance_active_, ltl_max_search_distance_, params_.bounding_radius,
                                ltl_step_size_, ltl_routers_, ltl_routers_active_, params_.lambda, ltl_min_altitude_,
                                ltl_max_altitude_, ltl_min_altitude_active_, ltl_max_altitude_active_)){
        best_node_ = new_node; // in case no way home is found, the agent continues exploring
      }
      if (dist_to_home.norm() < 1.3*params_.extension_range)
      {
          best_node_ = new_node;
          ROS_INFO("stl_aeplanner, path home found");
          break;
      }
      if (n == max_it -1) 
          ROS_ERROR_STREAM("No path home found");
  }
}

void STLAEPlanner::expandRRTUfomap(value_rtree* rtree,
                                 std::shared_ptr<point_rtree> stl_rtree,
                                 const Eigen::Vector4d& current_state)
{
  // Expand an RRT tree and calculate information gain in every node
  for (int n = 0;
       (n < params_.init_iterations or
        (n < params_.cutoff_iterations and
         best_node_->score(stl_rtree, ltl_lambda_, ltl_min_distance_, ltl_max_distance_, ltl_min_distance_active_,
                           ltl_max_distance_active_, ltl_max_search_distance_, params_.bounding_radius, ltl_step_size_,
                           ltl_routers_, ltl_routers_active_, params_.lambda, ltl_min_altitude_, ltl_max_altitude_,
                           ltl_min_altitude_active_, ltl_max_altitude_active_) < params_.zero_gain)) and
       ros::ok();
       ++n)
  {
    std::shared_ptr<RRTNode> new_node = std::make_shared<RRTNode>();
    std::shared_ptr<RRTNode> nearest;
    const ufomap::OccupancyNode* ot_result;

    // Sample new point around agent and check that
    // (1) it is within the boundaries
    // (2) it is in known space
    // (3) the path between the new node and it')s parent does not contain any
    // obstacles
    do
    {
      Eigen::Vector4d offset;
      offset = sampleNewPoint();
      new_node->state_ = current_state + offset;
      nearest = chooseParent(*rtree, stl_rtree, new_node, params_.extension_range);
      new_node->state_ = restrictDistance(nearest->state_, new_node->state_);
      //ufomap_mutex_.lock();
      ot_result = ufomap_.search(ufomap::Point3f(new_node->state_[0], new_node->state_[1], new_node->state_[2]));
      //ufomap_mutex_.unlock();
      if (ot_result == NULL){
        continue;
      }
    } while (!isInsideBoundaries(new_node->state_) or (!ot_result or !ufomap_.isFree(*ot_result)) or
             collisionLine(stl_rtree, nearest->state_, new_node->state_, params_.bounding_radius));

    // new_node is now ready to be added to tree
    new_node->parent_ = nearest;
    nearest->children_.push_back(new_node);

    // rewire tree with new node
    rewire(*rtree, stl_rtree, new_node, params_.extension_range, params_.bounding_radius, params_.d_overshoot_);

    // Calculate potential information gain for new_node
    std::pair<double, double> ret = getGain(new_node);
    new_node->state_[3] = ret.second;  // Assign yaw angle that maximizes g
    new_node->gain_ = ret.first;
    rtree->insert(std::make_pair(point(new_node->state_[0], new_node->state_[1], new_node->state_[2]), new_node));

    // Update best node
    if (!best_node_ or
        new_node->score(stl_rtree, ltl_lambda_, ltl_min_distance_, ltl_max_distance_, ltl_min_distance_active_,
                        ltl_max_distance_active_, ltl_max_search_distance_, params_.bounding_radius, ltl_step_size_,
                        ltl_routers_, ltl_routers_active_, params_.lambda, ltl_min_altitude_, ltl_max_altitude_,
                        ltl_min_altitude_active_, ltl_max_altitude_active_) >
            best_node_->score(stl_rtree, ltl_lambda_, ltl_min_distance_, ltl_max_distance_, ltl_min_distance_active_,
                              ltl_max_distance_active_, ltl_max_search_distance_, params_.bounding_radius,
                              ltl_step_size_, ltl_routers_, ltl_routers_active_, params_.lambda, ltl_min_altitude_,
                              ltl_max_altitude_, ltl_min_altitude_active_, ltl_max_altitude_active_)){
      best_node_ = new_node;
    }
  }
}

Eigen::Vector4d STLAEPlanner::sampleNewPoint()
{
  // Samples one point uniformly over a sphere with a radius of
  // param_.max_sampling_radius
  Eigen::Vector4d point(0.0, 0.0, 0.0, 0.0);
  do
  {
    for (int i = 0; i < 3; i++)
      point[i] = params_.max_sampling_radius * 2.0 * (((double)rand()) / ((double)RAND_MAX) - 0.5);
  } while (point.squaredNorm() > max_sampling_radius_squared_);

  return point;
}

std::shared_ptr<RRTNode> STLAEPlanner::chooseParent(const value_rtree& rtree, std::shared_ptr<point_rtree> stl_rtree,
                                                    std::shared_ptr<RRTNode> node, double l)
{
  // Find nearest neighbour
  // TODO: How many neighbours to look for?
  std::vector<value> nearest;
  point bbx_min(node->state_[0] - (2 * l), node->state_[1] - (2 * l), node->state_[2] - (2 * l));
  point bbx_max(node->state_[0] + (2 * l), node->state_[1] + (2 * l), node->state_[2] + (2 * l));
  box query_box(bbx_min, bbx_max);
  rtree.query(boost::geometry::index::intersects(query_box), std::back_inserter(nearest));

  if (nearest.empty())
  {
    rtree.query(boost::geometry::index::nearest(point(node->state_[0], node->state_[1], node->state_[2]), 15),
              std::back_inserter(nearest));
  }

  // TODO: Check if correct
  std::shared_ptr<RRTNode> best_node;
  double best_cost;

  for (value item : nearest)
  {
    std::shared_ptr<RRTNode> current_node = item.second;

    double current_cost = current_node->cost(
        stl_rtree, ltl_lambda_, ltl_min_distance_, ltl_max_distance_, ltl_min_distance_active_,
        ltl_max_distance_active_, ltl_max_search_distance_, params_.bounding_radius, ltl_step_size_, ltl_routers_,
        ltl_routers_active_, ltl_min_altitude_, ltl_max_altitude_, ltl_min_altitude_active_, ltl_max_altitude_active_);

    if (!best_node || current_cost < best_cost)
    {
      best_node = current_node;
      best_cost = current_cost;
    }
  }

  return best_node;
}

void STLAEPlanner::rewire(const value_rtree& rtree, std::shared_ptr<point_rtree> stl_rtree,
                          std::shared_ptr<RRTNode> new_node, double l, double r, double r_os)
{
  std::vector<value> nearest;
  point bbx_min(new_node->state_[0] - l, new_node->state_[1] - l, new_node->state_[2] - l);
  point bbx_max(new_node->state_[0] + l, new_node->state_[1] + l, new_node->state_[2] + l);
  box query_box(bbx_min, bbx_max);
  rtree.query(boost::geometry::index::intersects(query_box), std::back_inserter(nearest));

  Eigen::Vector3d p1(new_node->state_[0], new_node->state_[1], new_node->state_[2]);

  double new_cost = new_node->cost(
      stl_rtree, ltl_lambda_, ltl_min_distance_, ltl_max_distance_, ltl_min_distance_active_, ltl_max_distance_active_,
      ltl_max_search_distance_, params_.bounding_radius, ltl_step_size_, ltl_routers_, ltl_routers_active_,
      ltl_min_altitude_, ltl_max_altitude_, ltl_min_altitude_active_, ltl_max_altitude_active_);

  for (size_t i = 0; i < nearest.size(); ++i)
  {
    std::shared_ptr<RRTNode> current_node = nearest[i].second;
    
    if (current_node == root_ || current_node == new_node)
    {
      continue;
    }
    
    Eigen::Vector3d p2(current_node->state_[0], current_node->state_[1], current_node->state_[2]);

    if (current_node->cost(stl_rtree, ltl_lambda_, ltl_min_distance_, ltl_max_distance_, ltl_min_distance_active_,
                           ltl_max_distance_active_, ltl_max_search_distance_, params_.bounding_radius, ltl_step_size_,
                           ltl_routers_, ltl_routers_active_, ltl_min_altitude_, ltl_max_altitude_,
                           ltl_min_altitude_active_, ltl_max_altitude_active_) > new_cost + (p1 - p2).norm())
    {
      if (!collisionLine(stl_rtree, new_node->state_, current_node->state_, r))
      {
        current_node->parent_->children_.erase(std::remove(current_node->parent_->children_.begin(), current_node->parent_->children_.end(), current_node), current_node->parent_->children_.end());
        current_node->parent_ = new_node;
        new_node->children_.push_back(current_node);
      }
    }
  }
}

Eigen::Vector4d STLAEPlanner::restrictDistance(Eigen::Vector4d nearest, Eigen::Vector4d new_pos)
{
  // Check for collision
  Eigen::Vector3d origin(nearest[0], nearest[1], nearest[2]);
  Eigen::Vector3d direction(new_pos[0] - origin[0], new_pos[1] - origin[1], new_pos[2] - origin[2]);
  // if (direction.norm() > params_.extension_range)
  if (direction.norm() > params_.extension_range)
    direction = params_.extension_range * direction.normalized();

  new_pos[0] = origin[0] + direction[0];
  new_pos[1] = origin[1] + direction[1];
  new_pos[2] = origin[2] + direction[2];

  return new_pos;
}

std::pair<double, double> STLAEPlanner::getGain(std::shared_ptr<RRTNode> node)
{
  stl_aeplanner_msgs::Query srv;
  srv.request.point.x = node->state_[0];
  srv.request.point.y = node->state_[1];
  srv.request.point.z = node->state_[2];

  if (gp_query_client_.call(srv))
  {
    if (srv.response.sigma < params_.sigma_thresh) // TODO: Figure out what this is
    {
      double gain = srv.response.mu;
      double yaw = srv.response.yaw;
      return std::make_pair(gain, yaw);
    }
  }

  node->gain_explicitly_calculated_ = true;
  return gainCubatureUfomap(node->state_);
}

bool STLAEPlanner::reevaluate(stl_aeplanner_msgs::Reevaluate::Request& req,
                              stl_aeplanner_msgs::Reevaluate::Response& res)
{
  for (std::vector<geometry_msgs::Point>::iterator it = req.point.begin(); it != req.point.end(); ++it)
  {
    Eigen::Vector4d pos(it->x, it->y, it->z, 0);
    std::pair<double, double> gain_response = gainCubatureUfomap(pos);
    res.gain.push_back(gain_response.first);
    res.yaw.push_back(gain_response.second);
  }

  return true;
}

std::pair<double, double> STLAEPlanner::gainCubatureUfomap(Eigen::Vector4d state)
{
  double gain = 0.0;

  // This function computes the gain using ufomap
  double fov_y = params_.hfov, fov_p = params_.vfov;

  double dr = params_.dr, dphi = params_.dphi, dtheta = params_.dtheta;
  double dphi_rad = M_PI * dphi / 180.0f, dtheta_rad = M_PI * dtheta / 180.0f;
  double r;
  int phi, theta;
  double phi_rad, theta_rad;
  double dV = 0;
  double theta_bbx = 1*dtheta_rad * 1e-9;
  double phi_bbx = 1*dphi_rad * 1e-9;
  double r_bbx = 2*dr;

  std::map<int, double> gain_per_yaw;

  Eigen::Vector3d origin(state[0], state[1], state[2]);
  Eigen::Vector3d vec, dir;

  //ufomap_mutex_.lock();
  for (theta = -180; theta < 180; theta += dtheta)
  {
    theta_rad = M_PI * theta / 180.0f;
    for (phi = 90 - fov_p / 2; phi < 90 + fov_p / 2; phi += dphi)
    {
      phi_rad = M_PI * phi / 180.0f;

      double gain_yaw = 0;
      for (r = params_.r_min; r < params_.r_max; r += dr)
      {
        vec[0] = state[0] + r * cos(theta_rad) * sin(phi_rad);
        vec[1] = state[1] + r * sin(theta_rad) * sin(phi_rad);
        vec[2] = state[2] + r * cos(phi_rad);
        dir = vec - origin;

        ufomap::Point3f query(vec[0], vec[1], vec[2]);
        const ufomap::OccupancyNode* result = ufomap_.search(query);

        Eigen::Vector4d v(vec[0], vec[1], vec[2], 0);
        if (!isInsideBoundaries(v))
          break;
        if (result) {
          dV = (2 * r * r * dr + 1.0 / 6 * dr * dr * dr) * dtheta_rad * sin(phi_rad) * sin(dphi_rad / 2); // volume of volumeelement + some discretization term
            double r_infl = r + r_bbx;
            bool break_r = false;
            // inflation of occupied space to avoid counting dynamic gain due to noise close to occupied space, the iflation is conical and thus increases futher away from the agent
            for (double theta_infl =  theta_rad - theta_bbx; theta_infl <= (theta_rad + theta_bbx); theta_infl += dtheta_rad){
              for (double phi_infl =  phi_rad - phi_bbx; phi_infl <= (phi_rad + phi_bbx); phi_infl += dphi_rad){
                vec[0] = state[0] + r_infl * cos(theta_infl) * sin(phi_infl);
                vec[1] = state[1] + r_infl * sin(theta_infl) * sin(phi_infl);
                vec[2] = state[2] + r_infl * cos(phi_infl);

                ufomap::Point3f query(vec[0], vec[1], vec[2]);
                const ufomap::OccupancyNode* result_infl = ufomap_.search(query);
                if (result_infl){
                  if (ufomap_.isOccupied(*result_infl)){
                    gain_yaw += gain_function(&ufomap_, result_infl, session_number_, dV);
                    break_r = true;
                    break; // Break if occupied so we don't count any information gain behind or close to a wall.
                  }
                }
              }
              if (break_r){break;} // Break if occupied so we don't count any information gain behind or close to a wall.
            }
          if (break_r){break;} // Break if occupied so we don't count any information gain behind or close to a wall.
          gain_yaw += gain_function(&ufomap_, result, session_number_, dV);
        }
      }
      gain += gain_yaw;
      gain_per_yaw[theta] += gain_yaw;
    }
  }
  //ufomap_mutex_.unlock();
  int best_yaw = 0;
  double best_yaw_score = 0;
  for (int yaw = -180; yaw < 180; yaw++)
  {
    double yaw_score = 0;
    for (int fov = -fov_y / 2; fov < fov_y / 2; fov++)
    {
      int theta = yaw + fov;
      if (theta < -180)
        theta += 360;
      if (theta > 180)
        theta -= 360;
      yaw_score += gain_per_yaw[theta];
    }

    if (best_yaw_score < yaw_score)
    {
      best_yaw_score = yaw_score;
      best_yaw = yaw;
    }

  }
  double r_max = params_.r_max;
  double h_max = params_.hfov / M_PI * 180;
  double v_max = params_.vfov / M_PI * 180;

  gain = best_yaw_score;  // / ((r_max*r_max*r_max/3) * h_max * (1-cos(v_max))) ;
  // ROS_ERROR_STREAM(gain);

  double yaw = M_PI * best_yaw / 180.f;

  state[3] = yaw;
  return std::make_pair(gain, yaw);
}

double STLAEPlanner::gain_function(const ufomap::Octree* ufomap, const ufomap::OccupancyNode* node, int session_number, double dV = 1)  // Outside of class to in order to be accessible to ufomap_visualization
{
  double g_unknwn = 0;
  double g_dyn_occ = 0;
  double g_dyn_free = 0;
  double g_last_obs = 0;
  double gain = 0;
  if (ufomap->isOccupied(*node)){
    g_dyn_occ += node->p_exit * dV;
  }
  if (ufomap->isUnknown(*node)){
    g_unknwn += dV;
  }
  if (ufomap->isFree(*node)){  // Search if there are any occupied space close to this node, discard it if to minimize the impact of noise
    g_dyn_free += node->p_entry * dV;
  }
  g_last_obs += (session_number - node->session_last_seen) * dV;
  gain = g_unknwn + g_dyn_occ + 1.0/20* g_dyn_free + 1.0/20*1.0/20*1.0/7*g_last_obs; 
  return gain;
}

geometry_msgs::PoseArray STLAEPlanner::getFrontiers()
{
  geometry_msgs::PoseArray frontiers;

  stl_aeplanner_msgs::BestNode srv;
  srv.request.threshold = 0.75;
  if (best_node_client_.call(srv))
  {
    for (int i = 0; i < srv.response.best_node.size(); ++i)
    {
      geometry_msgs::Pose frontier;
      frontier.position = srv.response.best_node[i];
      frontiers.poses.push_back(frontier);
    }
  }
  else
  {
  }

  return frontiers;
}

bool STLAEPlanner::isInsideBoundaries(Eigen::Vector4d point)
{
  return point[0] > params_.boundary_min[0] and point[0] < params_.boundary_max[0] and
         point[1] > params_.boundary_min[1] and point[1] < params_.boundary_max[1] and
         point[2] > params_.boundary_min[2] and point[2] < params_.boundary_max[2];
}

bool STLAEPlanner::isInsideBoundaries(Eigen::Vector3d point)
{
  return point[0] > params_.boundary_min[0] and point[0] < params_.boundary_max[0] and
         point[1] > params_.boundary_min[1] and point[1] < params_.boundary_max[1] and
         point[2] > params_.boundary_min[2] and point[2] < params_.boundary_max[2];
}

//bool STLAEPlanner::isInsideBoundaries(octomap::point3d point)
//{
//  return point.x() > params_.boundary_min[0] and point.x() < params_.boundary_max[0] and
//         point.y() > params_.boundary_min[1] and point.y() < params_.boundary_max[1] and
//         point.z() > params_.boundary_min[2] and point.z() < params_.boundary_max[2];
//}
bool STLAEPlanner::isInsideBoundaries(ufomap::Point3f point)
{
  return point.x() > params_.boundary_min[0] and point.x() < params_.boundary_max[0] and
         point.y() > params_.boundary_min[1] and point.y() < params_.boundary_max[1] and
         point.z() > params_.boundary_min[2] and point.z() < params_.boundary_max[2];
}

bool STLAEPlanner::collisionLine(std::shared_ptr<point_rtree> stl_rtree, Eigen::Vector4d p1, Eigen::Vector4d p2,
                                 double r)
{
  ufomap::Point3f start(p1[0], p1[1], p1[2]);
  ufomap::Point3f end(p2[0], p2[1], p2[2]);

  point bbx_min(std::min(p1[0], p2[0]) - r, std::min(p1[1], p2[1]) - r, std::min(p1[2], p2[2]) - r);
  point bbx_max(std::max(p1[0], p2[0]) + r, std::max(p1[1], p2[1]) + r, std::max(p1[2], p2[2]) + r);

  box query_box(bbx_min, bbx_max);
  std::vector<point> hits;
  stl_rtree->query(boost::geometry::index::intersects(query_box), std::back_inserter(hits));

  double lsq = (end - start).squaredNorm();
  double rsq = r * r;

  for (size_t i = 0; i < hits.size(); ++i)
  {
    ufomap::Point3f pt(hits[i].get<0>(), hits[i].get<1>(), hits[i].get<2>());

    if (CylTest_CapsFirst(start, end, lsq, rsq, pt) > 0 or (end - pt).norm() < r)
    {
      return true;
    }
  }
  return false;
}

//void STLAEPlanner::octomapCallback(const octomap_msgs::Octomap& msg)
//{
//  octomap::AbstractOcTree* aot = octomap_msgs::msgToMap(msg);
//  octomap::OcTree* ot = (octomap::OcTree*)aot;
//  ot_ = std::make_shared<octomap::OcTree>(*ot);
//
//  delete ot;
//}

void STLAEPlanner::transformCallback(const geometry_msgs::TransformStamped::ConstPtr& msg)
{
    transform_ = *msg;  // TODO: Move this outside stl_aeplanner
}

void STLAEPlanner::ufomapCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud)
{
    ufomap::PointCloud ufo_cloud;
    geometry_msgs::TransformStamped transform;
    if (cloud->header.frame_id != "map") {
        const char* destination_frame = "map";
        const char* original_frame = cloud->header.frame_id.c_str();
        //ROS_INFO("destination frame  = %s", destination_frame);
        //ROS_INFO("original frame  = %s", original_frame);
      try {
        //bool cantf1 = tfListener_.waitForTransform(destination_frame, original_frame, cloud->header.stamp, ros::Duration(1));
        //ROS_INFO("waitForTransform");
        //bool canTransform = tfBuffer_.canTransform(destination_frame, original_frame, cloud->header.stamp);
        //ROS_INFO("can transform 2 %d",canTransform);
        //ROS_INFO("can transform 1 %d",cantf1);
        //std::string s = tfBuffer_.allFramesAsString();
        //ROS_INFO("All frames: %s", s.c_str());
        
        //tfBuffer_.waitForTransform("/base_link", "/map", ros::Time(0), ros::Duration(3.0));
        transform = tfBuffer_.lookupTransform(destination_frame, 
                original_frame,
                cloud->header.stamp );
        sensor_msgs::PointCloud2::Ptr transformed_cloud(new sensor_msgs::PointCloud2);
        //tfListener_.transformPointCloud("groda", *cloud, *transformed_cloud);
        //tfListener_.transformPointCloud(destination_frame, transform, cloud->header.stamp, cloud, transformed_cloud);
        tf2::doTransform(*cloud, *transformed_cloud, transform);
        ufomap::toUfomap(transformed_cloud, &ufo_cloud);
      }
        catch(tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
            return;
        }
    }
    else{
        ufomap::toUfomap(cloud, &ufo_cloud);
    }

    ufomap::Point3f sensor_origin(transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z);
    //ufomap_mutex_.lock();
    {
        const std::lock_guard<std::mutex> lock(ufomap_mutex_);
        //ROS_INFO("b4 insetion of pointcloud");
        ufomap_.insertPointCloud(ufo_cloud, sensor_origin, ufomap_max_range_, 0, false, 0); // if virtual_camera topic is used, make sure that max range is less than the topics max range  
        //ROS_INFO("between insertion and viz");
        ufomap_viz_.standard_visualization(&ufomap_, session_number_, STLAEPlanner::gain_function);
        //ROS_INFO("after viz");
    }
    //ufomap_mutex_.unlock();
}

void STLAEPlanner::publishEvaluatedNodesRecursive(std::shared_ptr<RRTNode> node)
{
  if (!node)
    return;
  for (typename std::vector<std::shared_ptr<RRTNode>>::iterator node_it = node->children_.begin();
       node_it != node->children_.end(); ++node_it)
  {
    if ((*node_it)->gain_explicitly_calculated_)
    {
      stl_aeplanner_msgs::Node pig_node;
      pig_node.gain = (*node_it)->gain_;
      pig_node.pose.pose.position.x = (*node_it)->state_[0];
      pig_node.pose.pose.position.y = (*node_it)->state_[1];
      pig_node.pose.pose.position.z = (*node_it)->state_[2];
      tf2::Quaternion q;
      q.setRPY(0, 0, (*node_it)->state_[3]);
      pig_node.pose.pose.orientation.x = q.x();
      pig_node.pose.pose.orientation.y = q.y();
      pig_node.pose.pose.orientation.z = q.z();
      pig_node.pose.pose.orientation.w = q.w();
      gain_pub_.publish(pig_node);
    }

    publishEvaluatedNodesRecursive(*node_it);
  }
}

void STLAEPlanner::agentPoseCallback(const geometry_msgs::PoseStamped& msg)
{
  current_state_[0] = msg.pose.position.x;
  current_state_[1] = msg.pose.position.y;
  current_state_[2] = msg.pose.position.z;
  current_state_[3] = tf2::getYaw(msg.pose.orientation);

  current_state_initialized_ = true;

  // LTL Path
  bool add_to_ltl_path = true;
  if (ltl_path_.poses.size() != 0)
  {
    Eigen::Vector3d last_state(ltl_path_.poses[ltl_path_.poses.size() - 1].pose.position.x,
                               ltl_path_.poses[ltl_path_.poses.size() - 1].pose.position.y,
                               ltl_path_.poses[ltl_path_.poses.size() - 1].pose.position.z);
    Eigen::Vector3d new_state(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);

    if ((last_state - new_state).norm() < ltl_dist_add_path_)
    {
      add_to_ltl_path = false;
    }
  }

  if (add_to_ltl_path)
  {
    ltl_path_.poses.push_back(msg);
    ltl_path_.header.frame_id = "map";
    ltl_path_.header.stamp = ros::Time::now();
    ltl_path_pub_.publish(ltl_path_);
  }

  // Stats
  if (0 < ufomap_.getNumLeafNodes())
  {
    stl_aeplanner_msgs::LTLStats ltl_stats;
    ltl_stats.header.stamp = ros::Time::now();
    ltl_stats.ltl_min_distance = (ltl_min_distance_active_) ? ltl_min_distance_ : -1;
    ltl_stats.ltl_max_distance = (ltl_max_distance_active_) ? ltl_max_distance_ : -1;
    ltl_stats.ltl_min_altitude = (ltl_min_altitude_active_) ? ltl_min_altitude_ : -1;
    ltl_stats.ltl_max_altitude = (ltl_max_altitude_active_) ? ltl_max_altitude_ : -1;

    Eigen::Vector3d position(current_state_[0], current_state_[1], current_state_[2]);

    //std::shared_ptr<octomap::OcTree> ot = ot_;

    ufomap::Point3d min(position[0] - ltl_max_search_distance_, position[1] - ltl_max_search_distance_,
                         position[2] - ltl_max_search_distance_);

    ufomap::Point3d max(position[0] + ltl_max_search_distance_, position[1] + ltl_max_search_distance_,
                         position[2] + ltl_max_search_distance_);

    std::shared_ptr<point_rtree> rtree = std::make_shared<point_rtree>(getRtreeUfomap(min, max));

    std::pair<double, double> closest_distance = RRTNode::getDistanceToClosestOccupiedBounded(
        rtree, position, position, ltl_max_search_distance_, params_.bounding_radius, ltl_step_size_);

    ltl_iterations_++;

    if (closest_distance.first >= ltl_max_search_distance_)
    {
      return;
    }

    ltl_stats.current_closest_distance = closest_distance.first;

    if (ltl_iterations_ == 1)
    {
      ltl_mean_closest_distance_ = closest_distance.first;
    }
    else
    {
      ltl_mean_closest_distance_ += (closest_distance.first - ltl_mean_closest_distance_) / ltl_iterations_;
    }

    ltl_stats.mean_closest_distance = ltl_mean_closest_distance_;

    ltl_stats.closest_router_distance =
        (ltl_routers_active_) ? RRTNode::getMaxRouterDifference(position, position, ltl_routers_, ltl_step_size_) : -1;

    // Altitude
    std::pair<double, double> closest_altitude = RRTNode::getAltitudeClosestOccupiedBounded(
        rtree, position, position, ltl_max_search_distance_, params_.bounding_radius, ltl_step_size_);

    if (closest_altitude.first >= ltl_max_search_distance_)
    {
      return;
    }

    ltl_stats.current_closest_altitude = closest_altitude.first;

    if (ltl_iterations_ == 1)
    {
      ltl_mean_closest_altitude_ = closest_altitude.first;
    }
    else
    {
      ltl_mean_closest_altitude_ += (closest_altitude.first - ltl_mean_closest_altitude_) / ltl_iterations_;
    }

    ltl_stats.mean_closest_altitude = ltl_mean_closest_altitude_;

    // Publish
    ltl_stats_pub_.publish(ltl_stats);
  }
}

geometry_msgs::Pose STLAEPlanner::vecToPose(Eigen::Vector4d state)
{
  tf::Vector3 origin(state[0], state[1], state[2]);
  double yaw = state[3];

  tf::Quaternion quat;
  quat.setEuler(0.0, 0.0, yaw);
  tf::Pose pose_tf(quat, origin);

  geometry_msgs::Pose pose;
  tf::poseTFToMsg(pose_tf, pose);

  return pose;
}

//-----------------------------------------------------------------------------
// Name: CylTest_CapsFirst
// Orig: Greg James - gjames@NVIDIA.com
// Lisc: Free code - no warranty & no money back.  Use it all you want
// Desc:
//    This function tests if the 3D point 'pt' lies within an arbitrarily
// oriented cylinder.  The cylinder is defined by an axis from 'pt1' to 'pt2',
// the axis having a length squared of 'lsq' (pre-compute for each cylinder
// to avoid repeated work!), and radius squared of 'rsq'.
//    The function tests against the end caps first, which is cheap -> only
// a single dot product to test against the parallel cylinder caps.  If the
// point is within these, more work is done to find the distance of the point
// from the cylinder axis.
//    Fancy Math (TM) makes the whole test possible with only two dot-products
// a subtract, and two multiplies.  For clarity, the 2nd mult is kept as a
// divide.  It might be faster to change this to a mult by also passing in
// 1/lengthsq and using that instead.
//    Elminiate the first 3 subtracts by specifying the cylinder as a base
// point on one end cap and a vector to the other end cap (pass in {dx,dy,dz}
// instead of 'pt2' ).
//
//    The dot product is constant along a plane perpendicular to a vector.
//    The magnitude of the cross product divided by one vector length is
// constant along a cylinder surface defined by the other vector as axis.
//
// Return:  -1.0 if point is outside the cylinder
// Return:  distance squared from cylinder axis if point is inside.
//
//-----------------------------------------------------------------------------
//float STLAEPlanner::CylTest_CapsFirst(const octomap::point3d& pt1, const octomap::point3d& pt2, float lsq, float rsq,
//                                      const octomap::point3d& pt)
//{
//  float dx, dy, dz;     // vector d  from line segment point 1 to point 2
//  float pdx, pdy, pdz;  // vector pd from point 1 to test point
//  float dot, dsq;
//
//  dx = pt2.x() - pt1.x();  // translate so pt1 is origin.  Make vector from
//  dy = pt2.y() - pt1.y();  // pt1 to pt2.  Need for this is easily eliminated
//  dz = pt2.z() - pt1.z();
//
//  pdx = pt.x() - pt1.x();  // vector from pt1 to test point.
//  pdy = pt.y() - pt1.y();
//  pdz = pt.z() - pt1.z();
//
//  // Dot the d and pd vectors to see if point lies behind the
//  // cylinder cap at pt1.x, pt1.y, pt1.z
//
//  dot = pdx * dx + pdy * dy + pdz * dz;
//
//  // If dot is less than zero the point is behind the pt1 cap.
//  // If greater than the cylinder axis line segment length squared
//  // then the point is outside the other end cap at pt2.
//
//  if (dot < 0.0f || dot > lsq)
//    return (-1.0f);
//  else
//  {
//    // Point lies within the parallel caps, so find
//    // distance squared from point to line, using the fact that sin^2 + cos^2 = 1
//    // the dot = cos() * |d||pd|, and cross*cross = sin^2 * |d|^2 * |pd|^2
//    // Carefull: '*' means mult for scalars and dotproduct for vectors
//    // In short, where dist is pt distance to cyl axis:
//    // dist = sin( pd to d ) * |pd|
//    // distsq = dsq = (1 - cos^2( pd to d)) * |pd|^2
//    // dsq = ( 1 - (pd * d)^2 / (|pd|^2 * |d|^2) ) * |pd|^2
//    // dsq = pd * pd - dot * dot / lengthsq
//    //  where lengthsq is d*d or |d|^2 that is passed into this function
//
//    // distance squared to the cylinder axis:
//
//    dsq = (pdx * pdx + pdy * pdy + pdz * pdz) - dot * dot / lsq;
//
//    if (dsq > rsq)
//      return (-1.0f);
//    else
//      return (dsq);  // return distance squared to axis
//  }
//}

float STLAEPlanner::CylTest_CapsFirst(const ufomap::Point3d& pt1, const ufomap::Point3d& pt2, float lsq, float rsq,
                                      const ufomap::Point3d& pt)
{
  float dx, dy, dz;     // vector d  from line segment point 1 to point 2
  float pdx, pdy, pdz;  // vector pd from point 1 to test point
  float dot, dsq;

  dx = pt2.x() - pt1.x();  // translate so pt1 is origin.  Make vector from
  dy = pt2.y() - pt1.y();  // pt1 to pt2.  Need for this is easily eliminated
  dz = pt2.z() - pt1.z();

  pdx = pt.x() - pt1.x();  // vector from pt1 to test point.
  pdy = pt.y() - pt1.y();
  pdz = pt.z() - pt1.z();

  // Dot the d and pd vectors to see if point lies behind the
  // cylinder cap at pt1.x, pt1.y, pt1.z

  dot = pdx * dx + pdy * dy + pdz * dz;

  // If dot is less than zero the point is behind the pt1 cap.
  // If greater than the cylinder axis line segment length squared
  // then the point is outside the other end cap at pt2.

  if (dot < 0.0f || dot > lsq)
    return (-1.0f);
  else
  {
    // Point lies within the parallel caps, so find
    // distance squared from point to line, using the fact that sin^2 + cos^2 = 1
    // the dot = cos() * |d||pd|, and cross*cross = sin^2 * |d|^2 * |pd|^2
    // Carefull: '*' means mult for scalars and dotproduct for vectors
    // In short, where dist is pt distance to cyl axis:
    // dist = sin( pd to d ) * |pd|
    // distsq = dsq = (1 - cos^2( pd to d)) * |pd|^2
    // dsq = ( 1 - (pd * d)^2 / (|pd|^2 * |d|^2) ) * |pd|^2
    // dsq = pd * pd - dot * dot / lengthsq
    //  where lengthsq is d*d or |d|^2 that is passed into this function

    // distance squared to the cylinder axis:

    dsq = (pdx * pdx + pdy * pdy + pdz * pdz) - dot * dot / lsq;

    if (dsq > rsq)
      return (-1.0f);
    else
      return (dsq);  // return distance squared to axis
  }
}

struct
{
  bool operator()(const std::pair<ufomap::Point3d, double>& lhs, const std::pair<ufomap::Point3d, double>& rhs) const
  {
    return lhs.second < rhs.second;
  }
} compareByDistance;

void STLAEPlanner::configCallback(stl_aeplanner::STLConfig& config, uint32_t level)
{
  ltl_lambda_ = config.lambda;
  ltl_min_distance_ = config.min_distance;
  ltl_max_distance_ = config.max_distance;
  ltl_min_distance_active_ = config.min_distance_active;
  ltl_max_distance_active_ = config.max_distance_active;
  ltl_routers_active_ = config.routers_active;
  ltl_dist_add_path_ = config.distance_add_path;
  ltl_max_search_distance_ = config.max_search_distance;
  ltl_step_size_ = config.step_size;

  ltl_min_altitude_ = config.min_altitude;
  ltl_max_altitude_ = config.max_altitude;
  ltl_min_altitude_active_ = config.min_altitude_active;
  ltl_max_altitude_active_ = config.max_altitude_active;
}

void STLAEPlanner::routerCallback(const dd_gazebo_plugins::Router::ConstPtr& msg)
{
  ltl_routers_[msg->id] = std::make_pair(msg->pose, msg->range);
}

}  // namespace stl_aeplanner
