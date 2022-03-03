using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReachTarget : MonoBehaviour
{
    public GameObject agent;
    public GameObject Arm;

    public void OnTriggerStay(Collider other){
        if(other.gameObject == Arm){
            agent.GetComponent<RobotAgent>().AddReward(0.01f);        
        }
    }
}
